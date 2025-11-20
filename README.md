Nima qiladi: raw_docx.jsonl dan instruction–response juftliklarini chiqaradi. Heuristic rejim offline va bepul, LLM-rejim aniqroq lekin API yoki local model kerak.
Ishga tushirish misoli:



# heuristic
python auto_extractor.py --in raw_docx.jsonl --out instr_pairs.jsonl --mode heuristic

# LLM-assisted (OpenAI): set OPENAI_API_KEY env or pass --openai_key
python auto_extractor.py --in raw_docx.jsonl --out instr_pairs.jsonl --mode llm --llm openai --openai_key YOUR_KEY
