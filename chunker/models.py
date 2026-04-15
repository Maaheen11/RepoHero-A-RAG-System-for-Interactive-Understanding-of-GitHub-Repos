## This file is only used to define the data structure for the chunker

from dataclasses import dataclass, field
from typing import List, Optional 

@dataclass
class Entity:
      # class, function, method

      type: str
      name: str
      signature: str

      start_line : int
      end_line : int

      docstring: Optional[str] = None 
      parent: Optional[str] = None

      children: List["Entity"] = field(default_factory=list)

@dataclass
class Chunk:
     # the output chunks

      file_path: str
      chunk_id: str

      start_line: int
      end_line: int

      text: str
      contextualized_text: str

      entities: List[Entity]
      scope_chain: List[str]

      imports: List[str]

      nws_count: int