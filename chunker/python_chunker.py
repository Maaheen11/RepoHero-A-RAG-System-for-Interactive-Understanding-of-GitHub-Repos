### Ref Post: https://supermemory.ai/blog/building-code-chunk-ast-aware-code-chunking/
### Ref Github: https://github.com/supermemoryai/code-chunk?ref=blog.supermemory.ai

"""
GitHub repo
      ↓
AST parser
      ↓
extract functions/classes/methods
      ↓
build scope trees
      ↓
package the text by entity(class/function/methods)
     ↓
create chunk
"""


"""
Optimal TODO:
1. _extract_annotation()
2. how to handle the ast.AST doesn't have end_lineno? -- current solution is to assume this entity's end_lineno == lineno: end_line=getattr(node, "end_lineno", node.lineno)
3. how to handle large entity without childre? -- current solution is to keep it as a chunk anyway
"""


import ast   # this built-in package is designed only for python files
from pathlib import Path
from typing import List, Optional

from .models import Entity, Chunk


class PythonASTChunker:
      def __init__(self, max_nws_chars: int = 12000):
            self.max_nws_chars = max_nws_chars
      
      def chunk_file(self, file_path: str):
            ## function for single file
            source = Path(file_path).read_text(encoding='utf-8')
            return self.chunk_source(file_path, source)
      
      def chunk_directory(self, folder_path:str):
            folder = Path(folder_path)
            chunks = []

            for file in folder.rglob("*.py"):  ## recursive all .py files in the folder
                  try:
                        chunks.extend(self.chunk_file(str(file)))
                  except FileNotFoundError as fe:
                        print(f"Error:{fe}")
                  except SyntaxError as se:
                        print(f"Error{se}")
            return chunks
      
      def chunk_source(self, file_path:str, source: str):
            tree = ast.parse(source)  # sourse is the file path
            lines = source.splitlines()  # source: Path()

            imports = self._extract_imports(tree)
            entities = self._extract_entities_with_parent(tree)
            root_entities = self._build_scope_tree(entities)

            chunks = self._greedy_chunk_entities(
                  file_path = file_path, 
                  lines = lines,
                  root_entities = root_entities, 
                  imports = imports
            )

            # if no extities were extracted, chunk the whole file
            if not chunks and lines:
                  text = "\n".join(lines)
                  contextualized_text = self._build_contextual_text(
                        file_path = file_path, 
                        text = text, 
                        entities = [],
                        imports = imports,
                        scope_chain = [], 
                  )

                  chunks.append(
                        Chunk(
                              file_path = file_path,
                              chunk_id = f"{Path(file_path).stem}_0",  # doesn't conclude the suffix of the file name
                              start_line = 1,
                              end_line = len(lines),
                              text = text,
                              contextualized_text = contextualized_text,
                              entities = [],
                              scope_chain = [],
                              imports = imports,
                              nws_count = self._count_nws(text),  
                        )
                  )

            return chunks
      
      def _count_nws(self, text: str):
            """
            count non-white-space characters
            """
            return sum(1 for ch in text if not ch.isspace())
      
      def _extract_imports(self, tree: ast.AST):
            imports = []

            for node in ast.walk(tree):
                  if isinstance(node, ast.Import):   # identify "import ..." lines
                        for alias in node.names:
                              imports.append(alias.name)
                  elif isinstance(node, ast.ImportFrom):  # identify "from ... import ..." lines
                        module = node.module or ""
                        for alias in node.names:
                              if module:
                                    imports.append(f"{module}.{alias.name}")
                              else:
                                    imports.append(alias.name)
            return imports 

      def _extract_entities_with_parent(self, tree: ast.AST):
            """
            Extract class/function/methods and preserve the parent info
            ast.walk(tree) will lose the hierachical ralationship between entities. 
            we want to know the function belong to which class. In another word, we need the parent node info. 
            """          

            entities = []


            def visit(node:ast.AST, parent_name: Optional[str]=None, in_class: bool = False):
                  if isinstance(node, ast.ClassDef):
                        entity = Entity(
                              type = "class",
                              name = node.name, 
                              signature=f"class {node.name}",
                              start_line = node.lineno,
                              end_line=getattr(node, "end_lineno", node.lineno), ## OPTIMAL2
                              docstring = ast.get_docstring(node),  # get annotation of the def, but cannot catch the annotation in other place 
                              parent=parent_name,
                        )
                        entities.append(entity)

                        for child  in node.body:
                              visit(child, parent_name=node.name, in_class=True)
                  
                  elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        entity_type = "method" if in_class else "function"
                        signature = self._build_function_signature(node)

                        entity = Entity(
                              type = entity_type,
                              name = node.name, 
                              signature=signature,
                              start_line = node.lineno,
                              end_line=getattr(node, "end_lineno", node.lineno),
                              docstring = ast.get_docstring(node), 
                              parent=parent_name,
                        )
                        entities.append(entity)

                        # nested function in function 
                        for child in node.body:
                              visit(child, parent_name=node.name, in_class=False)
                  else:
                        for child in ast.iter_child_nodes(node):
                              visit(child, parent_name=parent_name, in_class=in_class)
            visit(tree)
            entities.sort(key=lambda e: (e.start_line, e.end_line))
            return entities

      def _build_function_signature(self, node:ast.AST):
            """
            siganature: the stating part of the function, such as the name, type of variables
            """
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                  return ""
            
            args: list[str] = []
            for arg in node.args.args:
                  args.append(arg.arg)
            if node.args.vararg:
                  args.append("*"+node.args.vararg.arg)
            for arg in node.args.kwonlyargs:
                  args.append(arg.arg)
            if node.args.kwarg:
                  args.append("**"+node.args.kwarg.arg)
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            return f"{prefix} {node.name}({', '.join(args)})"
                                                                  
      def _build_scope_tree(self, entities:list[Entity]):
            """
            example: 
            ROOT
            ├── class UserService
            │     ├── method __init__
            │     └── method get_user
            │            └── function format_user
            │
            └── function add

            AST return the flat strcuture like: [class UserService, def __init__, def get_user, def format_user, def add]
            This function helps us to build tree structure by indicating the parent and child node.
            """

            roots: list[Entity] = []

            for entity in entities:
                  parent = self._find_deepest_parent(roots,entity)
                  if parent is not None:
                        parent.children.append(entity)
                  else:
                        roots.append(entity)
            return roots 

      def _find_deepest_parent(self, roots:list[Entity], target: Entity):
            def conatins(parent: Entity, child:Entity):
                  return (
                        parent.start_line <= child.start_line
                        and parent.end_line >= child.end_line
                        and not(
                              parent.start_line == child.start_line
                              and parent.end_line == child.end_line
                        )
                  )
            
            def dfs(node: Entity):
                  if not conatins(node, target):
                        return None 
                  
                  for child in node.children:
                        deeper = dfs(child)
                        if deeper is not None:
                              return deeper
                  return node

            for root in roots:
                  found = dfs(root)
                  if found is not None:
                        return found
            return None
      
      ### Chunking start

      def _greedy_chunk_entities(
                  self, 
                  file_path: str, 
                  lines: List[str], 
                  root_entities: List[Entity],
                  imports: List[str], 
      ):
            """
            group semantic entities into chunks.
            steps:
            1. keep each entity as a whole 
            2. if it is too large and has children, recurse into children
            3. if it is too large and no children, keep it as a single chunk ??? -- not sure if it is the best strategy [OPTIMAL3]
            """
            windows = self._pack_entities(root_entities, lines)
            chunks = []

            for idx, window in enumerate(windows):
                  start_line = min(entity.start_line for entity in window)
                  end_line = max(entity.end_line for entity in window)

                  text = "\n".join(lines[start_line-1:end_line])

                  first_entity = window[0]
                  scope_chain = self._infer_scope_chain(first_entity, root_entities)

                  contextualized_text = self._build_contextual_text(
                        file_path = file_path, 
                        text = text, 
                        entities = window, 
                        imports = imports, 
                        scope_chain = scope_chain,
                  )

                  chunks.append(
                        Chunk(
                              file_path = file_path,
                              chunk_id = f"{Path(file_path).stem}_{idx}",  # doesn't conclude the suffix of the file name
                              start_line = start_line,
                              end_line = end_line,
                              text = text,
                              contextualized_text = contextualized_text,
                              entities = window,
                              scope_chain = scope_chain,
                              imports = imports,
                              nws_count = self._count_nws(text),  
                        )
                  )

            return chunks
            
      def _pack_entities(self, entities: List[Entity], lines: List[str]):
            """
            pack entities into windows by chunking max_nws_chars
            """
            
            windows: List[List[Entity]] = []
            current_window: List[Entity] = []
            current_size = 0

            for entity in entities:
                  entity_text = '\n'.join(lines[entity.start_line-1:entity.end_line])
                  entity_size = self._count_nws(entity_text)

                  if current_size + entity_size <= self.max_nws_chars:
                        current_window.append(entity)
                        entity_size += entity_size
                  elif entity_size > self.max_nws_chars:
                        if current_window:
                              windows.append(current_window)
                              current_window = []
                              current_size = 0
                        
                        # large entity with children
                        if entity.children:
                              child_windows = self._pack_entities(entity.children, lines)
                              windows.extend(child_windows)
                        # no children
                        else:
                              windows.append([entity])
                  else:
                        if current_window:
                              windows.append(current_window)
                        
                        current_window = [entity]
                        current_size = entity_size
            
            if current_window:
                  windows.append(current_window)
            
            return windows
            
      def _build_contextual_text(
                  self, 
                  file_path: str, 
                  text: str, 
                  entities: List[Entity], 
                  imports: List[str], 
                  scope_chain: List[str], 
      ):
            """
            building a contextual version of the chunk for thr embedding
            contextualized_text = metadata + original_code
            """

            parts: List[str] = [f"# File: {file_path}"]

            """
            scope_chain: the path of entity in the code. such as UserService > get_user > format_user
            """
            if scope_chain:
                  parts.append(f"# Scope: {' > '.join(scope_chain)}")
            
            if entities:
                  signatures = [entity.signature for entity in entities if entity.signature]
                  if signatures:
                        parts.append(f"# Defines: {', '.join(signatures)}")
            
            if imports:
                  parts.append(f"# Imports: {', '.join(imports[:10])}")
            
            parts.append("")
            parts.append(text)

            return "\n".join(parts)
            
      def _infer_scope_chain(self, entity: Entity, roots: List[Entity]):
            chain = []
            current_parent = entity.parent
            chain.append(entity.name)

            while current_parent:
                  chain.append(current_parent)
                  parent_entity = self._find_entity_by_name(roots, current_parent)
                  current_parent = parent_entity.parent if parent_entity else None
            
            chain.reverse()
            return chain

      def _find_entity_by_name(self,roots: List[Entity], name: str):
            for root in roots:
                  found = self._dfs_find(root, name)
                  if found:
                        return found 
            return None
            
      def _dfs_find(self, node: Entity, name: str):
            if node.name == name:
                  return node
            
            for child in node.children:
                  found = self._dfs_find(child,name)
                  if found is not None:
                        return found
            return None 
                  

if __name__== "main":
      folder_path = "data"
      chunker = PythonASTChunker()
      chunks = chunker.chunk_directory(folder_path)
      print(chunks[0].contextualized_text)