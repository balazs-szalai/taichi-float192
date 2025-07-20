# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 21:41:49 2025

@author: balazs
"""

import ast
import inspect
import textwrap
import tempfile
import importlib.util
from typing import get_type_hints
import float192 as ty

def resolve_annotation(anotation_node, globals_dict):
    if isinstance(anotation_node, ast.Name): #example: f192_t
        return eval(anotation_node.id, globals_dict)
    elif isinstance(anotation_node, ast.Call): #example: ti.types.ndarray(dtype=f192_t, ndim = 3)
        src = ast.unparse(anotation_node)
        return eval(src, globals_dict).dtype
    elif isinstance(anotation_node, ast.Attribute): #example: float192.f192_t
        src = ast.unparse(anotation_node)
        return eval(src, globals_dict)
    return None

def replace_type_annotation(node, replacements):
    """Recursively replace known type annotations (AST) using replacements dict."""
    if isinstance(node, ast.Name) and node.id in replacements:
        return replacements[node.id]
    
    if isinstance(node, ast.Attribute) and node.attr in replacements:
        return replacements[node.attr]

    if isinstance(node, ast.Call):
        new_func = replace_type_annotation(node.func, replacements)
        new_args = [replace_type_annotation(arg, replacements) for arg in node.args]
        return ast.Call(func=new_func, args=new_args, keywords=node.keywords)

    return node


class TypeAnnotator(ast.NodeVisitor):
    def __init__(self, globals_dict=None):
        self.env = {}  # variable name -> type descriptor
        self.globals_dict = globals_dict if globals_dict is not None else globals()
        self.known_return_types = {
            'f192': ty.f192_t
        }

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            if arg.annotation:
                t = resolve_annotation(arg.annotation, self.globals_dict)
                self.env[arg.arg] = t
        
        if node.returns:
            node.inferred_type = resolve_annotation(node.returns, self.globals_dict)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            t = resolve_annotation(node.annotation, self.globals_dict)
            self.env[var_name] = t
            node.inferred_type = t
        self.generic_visit(node)

    def visit_Assign(self, node):
        self.visit(node.value)
        inferred = getattr(node.value, 'inferred_type', None)
        for target in node.targets:
            if isinstance(target, ast.Name) and inferred:
                self.env[target.id] = inferred
                target.inferred_type = inferred

    def visit_Name(self, node):
        t = self.env.get(node.id)
        if t:
            node.inferred_type = t

    def visit_Call(self, node):
        self.generic_visit(node)

        func_id = None
        if isinstance(node.func, ast.Name):
            func_id = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_id = ast.unparse(node.func)

        # Case 1: Known hardcoded return types
        if func_id in self.known_return_types:
            node.inferred_type = self.known_return_types[func_id]
            return node

        # Case 2: Use get_type_hints
        func_obj = eval(ast.unparse(node.func), self.globals_dict)
        if func_obj is not None:
            try:
                hints = get_type_hints(func_obj, globalns=self.globals_dict)
                ret_type = hints.get('return')
                if ret_type:
                    node.inferred_type = ret_type
            except Exception:
                pass

        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name):
            base = node.value.id
            node.inferred_type = self.env.get(base)

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)
        ltype = getattr(node.left, 'inferred_type', None)
        rtype = getattr(node.right, 'inferred_type', None)
        if ltype == rtype and ltype is not None:
            node.inferred_type = ltype

class BinOpTransformer(ast.NodeTransformer):
    def __init__(self, target_type, op_map):
        """
        target_type: the type (like U256()) to match against.
        op_map: mapping of ast operator class → replacement function name.
        e.g., {ast.Add: 'add_u256'}
        """
        self.target_type = target_type
        self.op_map = op_map

    def visit_BinOp(self, node):
        self.generic_visit(node)

        ltype = getattr(node.left, 'inferred_type', None)
        rtype = getattr(node.right, 'inferred_type', None)
        op_type = type(node.op)

        if ltype == self.target_type and rtype == self.target_type and op_type in self.op_map:
            func_name = self.op_map[op_type].__name__

            new_node = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='ty', ctx=ast.Load()),
                    attr=func_name,
                    ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[]
            )
            new_node.inferred_type = self.target_type
            return new_node

        return node

class AnnotationTransformer(ast.NodeTransformer):
    def __init__(self, replacements):
        # Dict like: {'U256': ast.parse("ti.types.vector(8, ti.u32)").body[0].value}
        self.replacements = replacements

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            if arg.annotation:
                arg.annotation = replace_type_annotation(arg.annotation, self.replacements)
        if node.returns:
            node.returns = replace_type_annotation(node.returns, self.replacements)
        return node

    def visit_AnnAssign(self, node):
        if node.annotation:
            node.annotation = replace_type_annotation(node.annotation, self.replacements)
        return node

class LoopDepthAnnotator(ast.NodeVisitor):
    def __init__(self):
        self.loop_depth = 0

    def generic_visit(self, node):
        node.loop_depth = self.loop_depth  # annotate every node
        super().generic_visit(node)

    def visit_For(self, node):
        node.loop_depth = self.loop_depth
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    # Optional: explicitly mark while loops as non-parallel
    def visit_While(self, node):
        node.loop_depth = self.loop_depth
        self.generic_visit(node)

def remove_decorator(decorator_list, deco_str):
    new_list = []
    for node in decorator_list:
        node_name = ast.unparse(node)
        if not node_name == deco_str:
            new_list.append(node)
    return new_list



def supports_f192(globals_dict):
    def supports_f192_base(fn):
        transformer = BinOpTransformer(ty.f192_t, {ast.Add: ty.add_f192,
                                                   ast.Sub: ty.sub_f192,
                                                   ast.Mult: ty.mul_f192,
                                                   ast.Div: ty.div_f192})
        annotator = TypeAnnotator(globals_dict)
        ann_swap = AnnotationTransformer({'f192_t': ast.parse("ti.types.vector(6, ti.u32)").body[0].value})
        
        source = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(source)
        annotator.visit(tree)
        ann_swap.visit(tree)
        
        tree.body[0].decorator_list = []

        # Transform function (annotations, ops, etc.)
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)

        # Get unparsed Python code
        transformed_source = ast.unparse(tree)
        
        file_path = '\\'.join(__file__.split('\\')[:-1])
        imports = f'''
from __main__ import *
import sys
sys.path.append(r'{file_path}')
import float192 as ty

'''

        # Write to a temporary .py file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(imports + transformed_source)
            temp_path = f.name

        # Load as Python module
        spec = importlib.util.spec_from_file_location("temp_module", temp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        ret_fnc = getattr(module, fn.__name__)
        
        return ret_fnc
    return supports_f192_base