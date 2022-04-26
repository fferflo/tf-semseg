import sympy, types, math, inspect
import tensorflow as tf
import numpy as np
from collections import defaultdict

def split(desc):
    elements = [""]
    nested_parentheses = 0
    for c in desc:
        if nested_parentheses == 0 and c == "(":
            nested_parentheses += 1
            elements[-1] += c
        elif nested_parentheses > 0 and c == ")":
            elements[-1] += c
            nested_parentheses -= 1
        elif nested_parentheses == 0 and c == " ":
            elements = elements + [""]
        else:
            elements[-1] += c
        if nested_parentheses < 0:
            raise ValueError("Invalid parentheses")
    if nested_parentheses != 0:
        raise ValueError("Invalid parentheses")

    return [e.strip() for e in elements if len(e) > 0]

def inferred_value(n):
    if isinstance(n, int):
        return n
    elif "__dict__" in dir(n) and "_inferred_value" in vars(n) and not n._inferred_value is None and not n._inferred_value[0] is None:
        return int(n._inferred_value[0])
    else:
        return None

class Value:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        value = inferred_value(self.value)
        if not value is None:
            return str(value)
        else:
            return self.name

    def get_ordered_values(self):
        return [self]

    def get_value(self):
        value = inferred_value(self.value)
        return self.value if value is None else self.value

class Variable:
    def __init__(self, name, context, ellipses=None):
        if name.isnumeric():
            self.value = int(name)
            self.name = f"__constantdim{context.constant_name_counter}({self.value})"
            context.constant_name_counter += 1
        else:
            self.value = None
            self.name = name
        self.basename = self.name
        self.ellipses = ellipses

    def __str__(self):
        return self.name

    def get_all(self):
        return [self]

    def sympy_variable_num_occurrence(self, name):
        return 1 if name == self.name else 0

    def sympy_total_num_occurrence(self):
        return 1

    def sympy_element_num(self):
        return 1

    def expand(self, context, values, ellipses=[]):
        return [EllipsisVariable(self, ellipses, context) if len(ellipses) > 0 else self]

    def fill(self, values):
        return Value(self.name, values[self.name])

class EllipsisVariable:
    def __init__(self, variable, ellipses, context):
        self.basename = variable.name
        self.value = variable.value
        self.name = self.basename + "-" + "-".join([str(i) for _, _, i in ellipses])
        self.ellipses = ellipses

    def __str__(self):
        return self.name

    def get_all(self):
        return [self]

    def constant_shape(self):
        return np.asarray([d for _, d, _ in self.ellipses])

    def fill(self, values):
        return Value(self.name, values[self.name])

class Ellipsis:
    def __init__(self, sub, context):
        self.sub = sub
        self.sympy_expansion_count = sympy.Symbol(f"__ellipsis{context.ellipsis_name_counter}", integer=True)
        context.ellipsis_name_counter += 1

    def __str__(self):
        return str(self.sub) + "..."

    def get_all(self):
        return [self] + self.sub.get_all()

    def sympy_variable_num_occurrence(self, name):
        return self.sympy_expansion_count * self.sub.sympy_variable_num_occurrence(name)

    def sympy_total_num_occurrence(self):
        return self.sympy_expansion_count * self.sub.sympy_total_num_occurrence()

    def sympy_element_num(self):
        return self.sympy_expansion_count

    def expand(self, context, values, ellipses=[]):
        expansion_count = values[self.sympy_expansion_count.name]
        return [x for i in range(expansion_count) for x in self.sub.expand(context, values, ellipses + [(self, expansion_count, i)])]

class Group:
    def __init__(self, children):
        self.children = children

    def __str__(self):
        return "(" + " ".join([str(c) for c in self.children]) + ")"

    def get_all(self):
        return [self] + [x for c in self.children for x in c.get_all()]

    def sympy_variable_num_occurrence(self, name):
        return sum(c.sympy_variable_num_occurrence(name) for c in self.children)

    def sympy_total_num_occurrence(self):
        return sum(c.sympy_total_num_occurrence() for c in self.children)

    def sympy_element_num(self):
        return 1

    def expand(self, context, values, ellipses=[]):
        children = []
        for c in self.children:
            children.extend(c.expand(context, values, ellipses))
        return [Group(children)]

    def fill(self, values):
        return Group([c.fill(values) for c in self.children])

    def get_ordered_values(self):
        return [x for c in self.children for x in c.get_ordered_values()]

    def get_value(self):
        return math.prod([c.value for c in self.children])

class Root(Group):
    def __init__(self, children):
        Group.__init__(self, children)

    def __str__(self):
        return " ".join([str(c) for c in self.children])

    def sympy_root_children_num(self):
        return sum(c.sympy_element_num() for c in self.children)

    def expand(self, context, values):
        children = []
        for c in self.children:
            children.extend(c.expand(context, values))
        return Root(children)

    def fill(self, values):
        return Root([c.fill(values) for c in self.children])

def parse_desc(desc, context, ellipses=[], root=True):
    desc = split(desc)
    if len(desc) == 1:
        desc = desc[0]
        if desc.endswith("..."):
            ellipsis = Ellipsis.__new__(Ellipsis)
            sub = parse_desc(desc[:-3], context=context, ellipses=ellipses + [ellipsis], root=False)
            assert len(sub) == 1
            ellipsis.__init__(sub[0], context=context)
            result = [ellipsis]
        elif desc[0] == "(":
            assert desc[-1] == ")"
            result = [Group(parse_desc(desc[1:-1], context=context, ellipses=ellipses, root=False))]
        else:
            assert not "(" in desc and not ")" in desc
            result = [Variable(name=desc, context=context, ellipses=ellipses)]
    else:
        result = [x for d in desc for x in parse_desc(d, context=context, ellipses=ellipses, root=False)]

    if root:
        result = Root(result)
    return result

cache = {}
def apply(desc, *tensors, reduction=None, output_shape=None, output_ndims=None, keepdims=False, **constants):
    tensors = list(tensors)
    input_shapes = []
    for tensor in tensors:
        tf_input_shape = None
        input_shape = []
        for j in range(len(tensor.shape)):
            static_dim = tensor.shape[j]
            if not static_dim is None:
                input_shape.append(static_dim)
            else:
                if tf_input_shape is None:
                    tf_input_shape = tf.shape(tensor)
                input_shape.append(tf_input_shape[j])
        input_shapes.append(input_shape)

    output_shape_len = None if output_shape is None else (output_shape.shape[0] if "shape" in dir(output_shape) else len(output_shape))
    def get_constants(descs_in, desc_out):
        result = []

        name_to_variables = defaultdict(list)
        all_variables = []
        for v in [v for l in ([desc_in.get_all() for desc_in in descs_in] + [desc_out.get_all()]) for v in l]:
            if isinstance(v, EllipsisVariable) or isinstance(v, Variable):
                name_to_variables[v.basename].append(v)
                all_variables.append(v)

        # Input shape
        for i, desc_in in enumerate(descs_in):
            for j, c in enumerate(desc_in.children):
                result.append((f"__input{i}shape{j}", c, input_shapes[i][j]))

        # Output shape if given
        if not output_shape is None:
            for i, c in enumerate(desc_out.children):
                result.append((f"__outputshape{i}", c, output_shape[i]))

        # Constant arguments if given
        for name, value in constants.items():
            value_broadcasted = None
            for variable in name_to_variables[name]:
                if isinstance(variable, EllipsisVariable):
                    dest_shape = variable.constant_shape()
                    if value_broadcasted is None:
                        value_broadcasted = tf.broadcast_to(value, dest_shape)
                    index = [int(i) for i in variable.name.split("-")[-dest_shape.shape[0]:]]

                    result.append((f"__constantarg-{variable.name}", variable, value_broadcasted[index]))
                else:
                    result.append((f"__constantarg-{variable.name}", variable, value))

        # Constant dimensions if given
        for variable in all_variables:
            if not variable.value is None:
                result.append((f"__constantdim-{variable.name}", variable, variable.value))

        return result

    cache_key = (
        tuple(len(input_shape) for input_shape in input_shapes),
        desc,
        reduction,
        output_shape_len if not output_shape is None else None,
        output_ndims,
        keepdims,
        tuple((k, tuple(tf.convert_to_tensor(constants[k]).shape.as_list())) for k in sorted(list(constants.keys()))),
    )
    if not cache_key in cache:
        # ########################### Parse description ###########################
        desc = desc.split("->")
        if len(desc) != 2:
            raise ValueError("Description must contain exactly one '->'")
        descs_in, desc_out = desc
        descs_in = descs_in.split(",")
        if len(descs_in) != len(tensors):
            raise ValueError(f"Expected {len(descs_in)} input tensors, got {len(tensors)}")
        context = types.SimpleNamespace(
            ellipsis_name_counter=0,
            constant_name_counter=0,
        )
        descs_in = [parse_desc(desc_in, context=context) for desc_in in descs_in]
        desc_out = parse_desc(desc_out, context=context)

        # ########################### Expand ellipses ###########################
        descs = descs_in + [desc_out]
        variable_names = [set(v.name for v in desc.get_all() if isinstance(v, Variable)) for desc in descs]
        # variable_names_in = set(v.name for v in desc_in.get_all() if isinstance(v, Variable))
        # variable_names_out = set(v.name for v in desc_out.get_all() if isinstance(v, Variable))
        # sympy_variable_in_num_occurrences = {name: sympy.Symbol(f"__in-{name}", integer=True) for name in variable_names_in}
        # sympy_variable_out_num_occurrences = {name: sympy.Symbol(f"__out-{name}", integer=True) for name in variable_names_out}
        sympy_variable_num_occurrences = [{name: sympy.Symbol(f"__in-{name}", integer=True) for name in vn} for vn in variable_names]
        all_nodes = [x for desc in descs for x in desc.get_all()]

        equations = []

        # Inputs/ outputs from desc
        for desc, vn, svno in zip(descs, variable_names, sympy_variable_num_occurrences):
            for name in vn:
                equations.append(sympy.Eq(svno[name], desc.sympy_variable_num_occurrence(name)))
        for desc1, vn1, svno1 in zip(descs, variable_names, sympy_variable_num_occurrences):
            for desc2, vn2, svno2 in zip(descs, variable_names, sympy_variable_num_occurrences):
                if id(desc1) != id(desc2):
                    for name in vn1.intersection(vn2):
                        equations.append(sympy.Eq(svno1[name], svno2[name]))

        # Inputs/ outputs from tensors and parameters
        for i, desc_in in enumerate(descs_in):
            equations.append(sympy.Eq(desc_in.sympy_root_children_num(), len(input_shapes[i])))
        if not output_shape is None:
            equations.append(sympy.Eq(desc_out.sympy_root_children_num(), output_shape_len))
        if not output_ndims is None:
            equations.append(sympy.Eq(desc_out.sympy_root_children_num(), output_ndims))

        # Constants
        name_to_any_variable = {v.name: v for v in all_nodes if isinstance(v, Variable)}
        for name, value in constants.items():
            ellipses = name_to_any_variable[name].ellipses
            expected_rank = len(name_to_any_variable[name].ellipses)
            def get_shape(value):
                if tf.is_tensor(value) or isinstance(value, np.ndarray):
                    return [int(i) for i in value.shape]
                elif isinstance(value, list) or isinstance(value, tuple):
                    if len(set(tuple(get_shape(v)) for v in value)) != 1:
                        raise ValueError(f"Invalid value for parameter {name}")
                    return [len(value)] + get_shape(value[0])
                else:
                    return []
            got_shape = get_shape(value)
            if expected_rank == len(got_shape):
                for i in range(len(got_shape)):
                    equations.append(sympy.Eq(ellipses[i].sympy_expansion_count, got_shape[i]))

        ellipses = [e for e in all_nodes if isinstance(e, Ellipsis)]
        variables = [e.sympy_expansion_count for e in ellipses] + [v for svno in sympy_variable_num_occurrences for v in svno.values()]
        equations2 = []
        for e in equations:
            try_equations = equations2 + [e]
            result = sympy.solve(try_equations, variables)
            if len(result) > 0:
                equations2 = try_equations
        equations = equations2

        expansion_values = sympy.solve(equations, variables)
        for k, v in expansion_values.items():
            if not v.is_number:
                k = str(k)
                if k.startswith("__ellipsis"):
                    raise ValueError(f"Failed to deduce expansion count of ellipsis {k[10:]}")
                else:
                    raise ValueError(f"Failed to deduce variable occurrence count of {k}")
        expansion_values = {str(k): int(v) for k, v in expansion_values.items()}

        descs = [desc.expand(context, expansion_values) for desc in descs]
        descs_in = descs[:-1]
        desc_out = descs[-1]

        # ########################### Find dimension values ###########################
        variable_names = [set(v.name for v in desc.get_all() if isinstance(v, Variable) or isinstance(v, EllipsisVariable)) for desc in descs]
        variable_names_in = variable_names[:-1]
        variable_names_out = variable_names[-1]
        sympy_variable_values = {name: sympy.Symbol(name, integer=True) for name in set(v for vn in variable_names for v in vn)}
        def sympy_value(n):
            if isinstance(n, Root) or isinstance(n, Group):
                return math.prod([sympy_value(c) for c in n.children])
            elif isinstance(n, EllipsisVariable) or isinstance(n, Variable):
                return sympy_variable_values[n.name]
            else:
                assert False

        equations = []
        sympy_constants = {}
        constants_values = {}
        # remaining_assertions = []
        for sympy_constant_name, node, value in get_constants(descs_in, desc_out):
            sympy_computed_value = sympy_value(node)
            sympy_constant = sympy.Symbol(sympy_constant_name, integer=True)
            equations_try = equations + [sympy.Eq(sympy_constant, sympy_computed_value)]
            if len(sympy.solve(equations_try, list(sympy_variable_values.values()))) > 0:
                equations = equations_try
                sympy_constants[sympy_constant_name] = sympy_constant
                constants_values[sympy_constant_name] = value
            # else:
            #     remaining_assertions.append((sympy_constant, sympy_computed_value, value))

        # Solve symbolic
        sympy_solved_variable_values = sympy.solve(equations, list(sympy_variable_values.values()))

        sympy_constants_names = list(sympy_constants.keys())
        sympy_constants = [sympy_constants[k] for k in sympy_constants_names]
        def to_func(expr, sympy_values):
            func = sympy.lambdify(sympy_values, expr)
            source = inspect.getsource(func)
            source = source.replace("/", "//")
            d = {}
            exec(source, d, d)
            func = d["_lambdifygenerated"]
            return func
        variable_funcs = {str(k): to_func(v, sympy_constants) for k, v in sympy_solved_variable_values.items()}

        cache[cache_key] = (tuple(descs), sympy_constants_names, variable_funcs, variable_names)
    else:
        descs, sympy_constants_names, variable_funcs, variable_names = cache[cache_key]
        descs_in = descs[:-1]
        desc_out = descs[-1]
        variable_names_in = variable_names[:-1]
        variable_names_out = variable_names[-1]

        constants_values = {}
        for sympy_constant_name, node, value in get_constants(descs_in, desc_out):
            constants_values[sympy_constant_name] = value

    # Compute shapes in tensorflow from solved equations
    constants_values = [constants_values[k] for k in sympy_constants_names]
    variable_values = {k: func(*constants_values) for k, func in variable_funcs.items()}

    descs = [desc.fill(variable_values) for desc in descs]
    descs_in = descs[:-1]
    desc_out = descs[-1]

    # TODO: make assertions work in keras and tf.function
    # sympy_variable_names = list(sympy_variable_values.keys())
    # sympy_variables = [sympy_variable_values[k] for k in sympy_variable_names]
    # variables = [variable_values[k] for k in sympy_variable_names]
    # for sympy_constant, sympy_computed_value, value in remaining_assertions:
    #     got = evaluate(sympy_computed_value, sympy_variables, variables)
    #     tf.debugging.assert_equal(value, got, f"Assertion failed for {sympy_constant.name}. Expected {value}, got {got}")

    # ########################### Transform tensor ###########################
    ordered_names = [[v.name for v in desc.get_ordered_values()] for desc in descs]
    ordered_values = [[v.value for v in desc.get_ordered_values()] for desc in descs]
    in_ordered_names = ordered_names[:-1]
    in_ordered_values = ordered_values[:-1]
    out_ordered_names = ordered_names[-1]
    out_ordered_values = ordered_values[-1]

    # Reshape nested input to flat input
    for i in range(len(tensors)):
        shape = in_ordered_values[i]
        if len(shape) != len(tensors[i].shape):
            # TODO: possibly two consecutive reshapes here and in reduction
            tensors[i] = tf.reshape(tensors[i], shape)

    # Reduce and squeeze input dimensions
    for i in range(len(tensors)):
        reduced_names = [name for name in variable_names[i] if not any(name in variable_names[j] for j in range(len(descs)) if i != j)]
        reduced_len1_names = set(name for name in reduced_names if inferred_value(variable_values[name]) == 1)
        if len(reduced_names) - len(reduced_len1_names) > 0 and reduction is None:
            raise ValueError("Expected reduction argument")
        if len(reduced_names) == 0 and not reduction is None:
            raise ValueError("No dimensions are reduced, but got reduction argument")
        if len(reduced_names) > 0:
            shape = [v for v, name in zip(in_ordered_values[i], in_ordered_names[i]) if not name in reduced_len1_names]
            if len(shape) != len(tensors[i].shape):
                tensors[i] = tf.reshape(tensors[i], shape)

            axes = sorted([in_ordered_names[i].index(name) for name in reduced_names if not name in reduced_len1_names])
            if len(axes) > 0:
                tf_reduce_name = f"reduce_{reduction}"
                if tf_reduce_name in vars(tf):
                    tensors[i] = vars(tf)[tf_reduce_name](tensors[i], axis=axes)
                else:
                    raise ValueError(f"Invalid reduction argument {reduction}")
    # Transform to flat output
    if len(descs_in) == 1:
        # Use tf.transpose for single inputs
        in_ordered_names_union = [n for n in in_ordered_names[0] if n in out_ordered_names]
        out_ordered_names_union = [n for n in out_ordered_names if n in in_ordered_names[0]]
        assert len(in_ordered_names_union) == len(tensors[0].shape)
        assert len(out_ordered_names_union) == len(tensors[0].shape)

        perm = []
        for out_dim in range(len(out_ordered_names_union)):
            out_name = out_ordered_names_union[out_dim]
            in_index = in_ordered_names_union.index(out_name)
            perm.append(in_index)
        if perm != list(range(len(perm))):
            tensors[0] = tf.transpose(tensors[0], perm)

        output_tensor = tensors[0]
    else:
        # Use tf.einsum for multiple inputs
        einsum_str = ""
        einsum_variables = {}
        def get_einsum_variable(name):
            if name in einsum_variables:
                return einsum_variables[name]
            else:
                v = chr(ord("a") + len(einsum_variables))
                if ord(v) > ord("z"):
                    raise ValueError(f"Only supports up to {ord('z') - ord('a') + 1} input tensors")
                einsum_variables[name] = v
                return v
        for i in range(len(ordered_names)):
            ordered_names_union = [name for name in ordered_names[i] if any(name in ordered_names[j] for j in range(len(ordered_names)) if i != j)]
            for name in ordered_names_union:
                einsum_str += get_einsum_variable(name) + " "
            if i < len(ordered_names) - 2:
                einsum_str += ", "
            elif i == len(ordered_names) - 2:
                einsum_str += " -> "

        output_tensor = tf.einsum(einsum_str, *tensors, optimize="optimal")

    # Expand and broadcast missing output dimensions
    broadcasted_names = [n for n in variable_names_out if not any(n in vn for vn in variable_names_in)]
    if len(broadcasted_names) > 0:
        slices = [(tf.newaxis if name in broadcasted_names else slice(None)) for name in out_ordered_names]
        output_tensor = output_tensor[slices]
        if not all(inferred_value(value) == 1 for name, value in zip(out_ordered_names, out_ordered_values) if name in broadcasted_names):
            output_tensor = tf.broadcast_to(output_tensor, out_ordered_values)

    # Reshape flat output to nested output
    shape = [v.get_value() for v in desc_out.children]
    if len(shape) != len(out_ordered_values):
        output_tensor = tf.reshape(output_tensor, shape)

    return output_tensor
