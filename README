This is a collection of data gathering and processing scripts designed to compare verified AI, suspect AI, and definitely human Wikipedia article's evaulations in Pangram, an AI detection tool.

## Files
* `get_test_wiki_pages.py` fetches random English Wikipedia articles from October 2022 or before, which cannot be AI generated. It also fetches random articles from [Category:Articles containing suspected AI-generated texts](https://en.wikipedia.org/wiki/Category:Articles_containing_suspected_AI-generated_texts) and [Category:AfC submissions declined as a large language model output](https://en.wikipedia.org/wiki/Category:AfC_submissions_declined_as_a_large_language_model_output). It dumps a list of those articles into three CSVs, and then outputs a cleaned version of the prose of those articles to folders in data/.

Usage: `python get_test_wiki_pages.py --n 25 --ai-n 25 --ai-drafts-n 25 --out wikipedia_sample.csv --ai-out wikipedia_ai_sample.csv --contact your_email@example.com`

* `generate_ai_articles.py` uses OpenRouter to generate articles about the subjects in `wikipedia_sample.csv` using Claude, Gemini, and ChatGPT, cleans the formatting to match the cleaning used on the real wikipedia articles, and writes them to the corresponding folders in data/ and metadata to `generated_articles.csv`.

Usage: `python generate_ai_articles.py --limit 10 --contact your_email@example.com`

* `run_pangram.py` runs the articles in data/ through Pangram, an AI detection tool, and outputs `pangram_results.csv`.

Usage: `python run_pangram.py --folder data/random_2022 --source-csv data/wikipedia_sample.csv`

* `summarize_pangram_results.py` outputs the percent in each category (AI/Human/Mixed) for each source (Random 2022 articles, random AI suspected articles, and our three LLMs).

Usage: `python summarize_pangram_results.py`

## Notes
This is very vibe coded. I've been sanity-checking it and verifying the outputs but code is almost entirely Claude Sonnet 5. Further Claude-generated documentation is in the docstring for each script; I have not necessarily checked it for accuracy.

To run, you will need an API key for Pangram and OpenRouter. Set them in the shell with `export PANGRAM_API_KEY=... ` and `export OPENROUTER_API_KEY=...`. Pangram runs are 5c/article or every 1,000 words, whichever is greater; OpenRouter prices vary but seems to be a couple cents an article. While I have tested and do not expect any problems there is NO GUARANTEE THAT THIS CODE WILL NOT EAT ALL YOUR TOKENS. Set sane limits and/or turn off auto-reload.

## Todo/known issues
* All AI-generated articles already have a Wikipedia article, which the AIs and/or Pangram can crib note from. Possible solution: use some topics from declined drafts as well.
* Something tagged/declined as AI may have been updated since then to remove the AI content. Do we want to grab the article rev as of the time of tagging?
* Currently we run alphabetically, which is not a random sample of our random sample if --limit < number of articles pangramed.
* Gemini and Claude model represetn the default free model of the chatbot (the likeliest model to use), for ChatGPT this was fairly confusing to determine so there might not be parity there.
* Look at G11/deleted? Unfortunately I don't want to run an adminbot and also there's attribution issues.
* Better handling of segments/mixed articles.

