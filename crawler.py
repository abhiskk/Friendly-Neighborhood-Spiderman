# -----------------------------------------------------------------------------
# Author: Abhishek Kadian (abhishekkadiyan@gmail.com)
# -----------------------------------------------------------------------------

import os
import logging
import wikipedia


def fetch(titlesfile, datafolder="data/"):
    """Reads list of titles from 'filepath', downloads wikipedia articles for
    titles and stores them as separate files in 'datafolder'.
    """
    if not os.path.exists(datafolder):
        os.makedirs(datafolder)
    with open(titlesfile, "r", encoding="utf-8") as f:
        for line in f:
            title = line.strip()
            logging.debug("Downloading \'{0}\'".format(title.encode("utf-8")))
            page = wikipedia.page(title)
            with open(datafolder + title + ".txt", "wb") as g:
                g.write(bytes(page.title + "\n", "utf-8"))
                g.write(bytes(page.content, "utf-8"))
