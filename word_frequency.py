from demo_data_module import get_demo_data, count_words
# basic imports
import pandas as pd
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import regex as re
from collections import Counter


# Word Frequency diagram
def frequency_diagram(df: pd.DataFrame, limit: int, output_loc: str):
    '''
    df: A pandas dataframe. The dataframe should be a word frequency dataframe with frequency as index
    limit: A numerical value indicating top n words
    output_loc: The output location of the plot. Should include <filename>.png
    '''
    ax = df.head(limit).plot(kind="barh", width=0.95, figsize=(8, limit//3))
    ax.invert_yaxis()
    ax.set(xlabel='Frequency', ylabel='Token', title='Top Words')
    fig = ax.get_figure()
    print(f"Saving figure to {output_loc}..")
    fig.savefig(output_loc)


# WordCloud diagram
def wordcloud(word_freq: pd.Series | pd.DataFrame, output_loc: str, title: str = None, max_words: int = 200, stopwords=None,
              figsize=(12, 12), wordcloud_style={"background_color": "black", "max_font_size": 150, "colormap": "Paired", "width": 800, "height": 400}):
    '''
    word_freq: A pandas Series or DataFrame containing the tokens and their frequencies
    output_loc: The output location of the plot. Should include <filename>.png
    title: A string representing the title of the plot
    max_words: A int representing the maximum number of words to include in the wordcloud
    stopwords: A list or iterator which contains stopwords
    '''
    wc = WordCloud(max_words=max_words, **wordcloud_style)

    if type(word_freq) == pd.Series:
        counter = Counter(word_freq.fillna(0).to_dict())
    else:
        counter = word_freq

    if stopwords is not None:
        counter = {token: freq for (
            token, freq) in counter.items() if token not in stopwords}

    wc.generate_from_frequencies(counter)

    plt.figure(figsize=figsize)
    plt.title(title)

    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    print(f"Saving wordcloud to {output_loc}..")
    plt.savefig(output_loc)


# Pie-chart
def pie_chart(frequencies: dict, output_loc: str, title: str = None):
    '''
    frequencies: A dictionary with keys as labels and corressponding values
    output_loc: The output location of the plot. Should include <filename>.png
    title: A string representing the title of the plot
    '''
    plt.figure()
    plt.pie(frequencies.values(), labels=frequencies.keys(),
            autopct=lambda p: f'{p:.2f}%, {p*sum(frequencies.values())/100 :.0f} ')
    if title:
        plt.legend(title=title)
    else:
        plt.legend()
    print(f"Saving plot to {output_loc}..")
    plt.savefig(output_loc)


# # Usage example

# df = count_words(get_demo_data())  # a demo token frequency DataFrame

# frequency_diagram(df, 15, "./top15.png")
# frequency_diagram(df, 10, "./top10.png")
# frequency_diagram(df, 20, "./top20.png")


# wordcloud(df["freq"], "./wordcloud_100.png",
#           title="Demo Wordcloud", max_words=100)
# wordcloud(df["freq"], "./wordcloud_50.png",
#           title="Demo Wordcloud", max_words=50)
# wordcloud(df["freq"], "./wordcloud_10.png",
#           title="Demo Wordcloud", max_words=10)


# pie_chart({"positive": 450, "negative": 450, "neutral": 100},
#           "./pie_plot.png", "Demo Pie Chart")
