# NLP 
## setup
1. `git clone git@gitlab.tubit.tu-berlin.de:NLP/nlp.git`
2. `pip install -r requirements.txt`  
  
# project topics: 
* [EmoContext-Codalab](https://competitions.codalab.org/competitions/19790); [EmoContext](https://www.humanizing-ai.com/emocontext.html)
* [Hyperpartisan News Detection](https://pan.webis.de/semeval19/semeval19-web/) -> dataset?

### [hatEval-2019](https://competitions.codalab.org/competitions/19935)
Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter  
data contains tweets like: `YOU HAVE GOT TO BE 💩IN’ ME 😳 The German Government Pays for 3 Week Vacation for Refugees to Go Home`   
here [someones github](https://github.com/rnjtsh/hatEval-2019)

### [OffensEval 2019](https://competitions.codalab.org/competitions/20011)
Identifying and Categorizing Offensive Language in Social Media (tweets)

## advanced stuff

### Transfer-Learning 

#### NLP using [flair](https://github.com/zalandoresearch/flair.git)
* flair only seems to know recurrent document-embedders, why no attention-based stuff?
* focus on transfer-learning, using pretrained ELMo-, BERT-,flair- Embeddings
* text-classification, sequence-tagging, relation classification?

#### Hierarchical Document Classification
[Hierarchical Attention Networks for Document Classification [Yang 2016]](https://aclweb.org/anthology/N16-1174)

#### finetuning a pretrained BERT
* [pytorchic-bert](https://github.com/dhlee347/pytorchic-bert.git)

### Machine Teaching

#### close/strong supervision
* can one do __active learning__ with [BRAT](http://brat.nlplab.org/) == [GaoleMeng](https://github.com/GaoleMeng/ActiveLearningAnnotationTool)?
* what is [tagtog](https://docs.tagtog.net/) ?
* [spacy prodigy](https://prodi.gy/) better/worse usability than BRAT?

#### [weak supervision](https://hazyresearch.github.io/snorkel/blog/ws_blog_post.html)
* [snorkel (data programming)](https://github.com/HazyResearch/snorkel)
  * how does snorkel learn which labeling functions are more accurate than others? -> via the generative model training phase, but how exactly does it work?
  * snorkel and __active learning__ [not yet tried by anyone?](https://github.com/HazyResearch/snorkel/issues/905) -> would need some feedback from the model which help the _data-progammer_ to write meaningful labeling-functions
