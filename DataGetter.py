import requests
from bs4 import BeautifulSoup

EvalSet_URL = "https://vesmir.cz/cz/on-line-clanky/2018/09/od-darwina-k-mimozemskym-civilizacim-ii.html"
DevSet_URL = "https://vesmir.cz/cz/casopis/archiv-casopisu/2018/cislo-10/jak-se-neztratit-mori.html"

URL = "https://vesmir.cz/cz/on-line-clanky/?page="
file = "VesmirCorpus.txt"


pages_num = 37 # number of pages with articles from vesmir.cz page
counter = 1
with open(file , "w", encoding="utf-8") as File:
    for n in range (1, pages_num+1):
        # get page number n
        request = requests.get(URL + str(n))
        soup = BeautifulSoup(request.content, "html.parser")
        for article in soup.find_all(name="div", class_="col-sm-9"):
            # All articles on page are marked with class_ = "col-sm-9" , iterate through them
            link_tag =  article.find("a") # get link to the article
            article_link = "https://vesmir.cz" + link_tag.get("href")
            headline = link_tag.text
            File.write(headline + "\n")

            article_request = requests.get(article_link)
            if article_link == EvalSet_URL or article_link == DevSet_URL: # skip dev and eval articles
                print("=============================================================") 
                continue
            article_soup = BeautifulSoup(article_request.content, "html.parser")

            article_tag = article_soup.find(name="div", class_="article")
            # write all paragraphs from the article to the Corpus and separate by new line
            for p in article_tag.find_all(name="p",recursive=False):
                payload = p.text.lower().strip()
                if payload == "":
                    pass
                else:
                    File.write(payload + '\n')
            print(str(counter) + ". Article with Headline:   " + headline[:30] + "   ADDED to corpus, URL: " + article_link)
            counter += 1
