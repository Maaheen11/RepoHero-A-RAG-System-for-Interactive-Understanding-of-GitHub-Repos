from main import RepoHero


if __name__ == "__main__":
      repo = RepoHero()

      chunks = repo.chunk(".\data\spellchk.py")
      print(chunks[0])


      q1 = "How many functions does this file have?"
      retrieved_knowledge = repo.retrieve(q1, top_k=5)

      for score, chunk in retrieved_knowledge:
            print(f"Chunk: {chunk}\nScore: {score}\n")
