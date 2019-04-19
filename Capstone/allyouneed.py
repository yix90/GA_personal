import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import scipy.stats as stats
import datetime as dt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
import nltk
from nltk.tokenize import word_tokenize

mbti = pd.read_pickle('mbti_ver2.pickle')
mbti_n = mbti.drop(['type','image','video_link','video_title',
                    'otherlink','emoticons','hashtag','mbti_ref','mbti_ref_most','ennea','tagged_words',
                   'tagged_words', 'is_E','is_S','is_T','is_J'], axis=1)
mbti_n['words_only'] = mbti_n['words_only'].apply(lambda x: ' '.join(x))
mbti_n['word_char_ratio'] = mbti_n['word_char_ratio'].fillna(np.median(mbti_n[~mbti_n['word_char_ratio'].isnull()]['word_char_ratio']))

y_E = mbti['is_E']
y_S = mbti['is_S']
y_T = mbti['is_T']
y_J = mbti['is_J']
X = mbti_n

class Thipe(object):

    def __init__(self, X, y, stan=True, rand=42, web=True, include_feature='all'):
        self.stan=stan
        self.web=web
        self.include_feature=include_feature
        self.X=X
        self.y=y
        self.random_state = rand
        self.columns=None
        self.X_train=None
        self.y_train=None
        #I will be porting some features to other classes sooooo...
        self.tfidf_list=[]
        self.tsvd_list=[]
        self.ss=None
        self.mms=None
        self.ch2=None
        self.sexy=None

    def baseline_acc(self):
        baseline = max(self.y.value_counts()[0], self.y.value_counts()[1]) / float(self.y.value_counts().sum())
        return baseline * 100,'\%'


    def trainy(self, testy=0.2, imbl=True):
        """
        I'll do the following here:
        1. Do train test split
        2. Convert X_train and X_test to DataFrame (to delete column later plus other purposes)
        3. Do tfidf using train section, use the model and fit the X_train and X_test (then can delete the wordchunk column)
        4. If StandardScaler, scale the training and test data. (Default = True)
        5. To prepare data for chi2 reduction we need to scale everything to above 0, so MinMaxScaler
        """

        #This is perhaps the main reason why this step is embedded in a class
        #Because the stratification would be different, everything would be different already, like the tfidf vocab for example
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=self.random_state,
                                                                      test_size=testy, stratify=self.y)

        self.y_train=y_train
        self.y_test=y_test

        X_train = pd.DataFrame(X_train, columns=self.columns)
        X_test = pd.DataFrame(X_test, columns=self.columns)
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        for i in np.arange(1,4):
            tfidf = TfidfVectorizer(stop_words='english',ngram_range=(i,i), decode_error='replace', max_features=10000)
            Xword_train = tfidf.fit_transform(X_train['words_only'])
            Xword_test = tfidf.transform(X_test['words_only'])

            #We need to reduce the size of the tfidf trained matrix first
            #But after running TruncatedSVD we cannot see the words specifically alr so too bad...
            tsvd = TruncatedSVD(n_components=500, algorithm='arpack', random_state=self.random_state)
            Xwordie_train = tsvd.fit_transform(Xword_train)
            Xwordie_test = tsvd.transform(Xword_test)
            Xwordie_train_df = pd.DataFrame(Xwordie_train,
                                            columns=[str(i)+'_'+str(b) for b in np.arange(1,Xwordie_train.shape[1]+1)])
            Xwordie_test_df = pd.DataFrame(Xwordie_test,
                                           columns=[str(i)+'_'+str(b) for b in np.arange(1,Xwordie_test.shape[1]+1)])
            df_train = pd.concat([df_train,Xwordie_train_df], axis=1)
            df_test = pd.concat([df_test,Xwordie_test_df], axis=1)
            self.tfidf_list.append(tfidf)
            self.tsvd_list.append(tsvd)

        X_train.drop(['words_only'],axis=1,inplace=True)
        X_test.drop(['words_only'],axis=1,inplace=True)
        X = self.X.drop(['words_only'], axis=1)
        if self.web:
            X_train.drop(['n_video','n_links','n_image','n_otherlink','mention_count','hashtag_count','mbti_ref_count','ennea_count',
                          'bracket_count'], axis=1, inplace=True)
            X_test.drop(['n_video','n_links','n_image','n_otherlink','mention_count','hashtag_count','mbti_ref_count','ennea_count',
                          'bracket_count'], axis=1, inplace=True)
            X.drop(['n_video','n_links','n_image','n_otherlink','mention_count','hashtag_count','mbti_ref_count','ennea_count',
                          'bracket_count'], axis=1, inplace=True)
        self.columns = X_train.columns

        #Standardization step
        if self.stan:
            ss = StandardScaler().fit(X)
            X_train = ss.transform(X_train)
            X_test = ss.transform(X_test)
            X_train = pd.DataFrame(X_train, columns=self.columns)
            X_test = pd.DataFrame(X_test, columns=self.columns)
            self.ss = ss

        #Join step
        if self.include_feature == 'words':
            X_train = df_train
            X_test = df_test
            columnie = X_train.columns
        else:
            X_train = X_train.join(df_train)
            X_test = X_test.join(df_test)
            columnie = X_train.columns


        #Scale again to between 0 and 1
        combined_X = pd.concat([X_train,X_test],axis=0)
        mms = MinMaxScaler().fit(combined_X)
        X_train = pd.DataFrame(mms.transform(X_train), columns=columnie)
        X_test = pd.DataFrame(mms.transform(X_test), columns=columnie)

        if imbl:
            imbler = RandomUnderSampler(random_state=42)
            X_train, y_train = imbler.fit_sample(X_train, y_train)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.mms = mms

        return X_train, X_test, y_train, y_test

    def reducy(self, tipe = 'chi2', n_features=100):
        """
        Use chi2 to shrink down features
        """
        ch2 = SelectKBest(chi2, k=n_features)
        X_train = ch2.fit_transform(self.X_train, self.y_train)
        X_test = ch2.transform(self.X_test)
        self.ch2=ch2
        self.X_train=X_train
        self.X_test=X_test
        return X_train, X_test


    def try_model(self, model, label):
        sexy = model.fit(self.X_train, self.y_train)
        print sexy.score(self.X_test, self.y_test)
        yhat = sexy.predict(X_test)
        self.sexy = sexy
        print classification_report(y_test, yhat, target_names=label)
        print confusion_matrix(y_test, yhat)

    def niceplot(self):
        pass

E = Thipe(X, y_E)
X_train, X_test, y_train, y_test = E.trainy()
X_train, X_test = E.reducy()
E.try_model(model=LogisticRegression(), label=['Introvert','Extrovert'])

S = Thipe(X, y_S)
X_train, X_test, y_train, y_test = S.trainy()
X_train, X_test = S.reducy()
S.try_model(model=LogisticRegression(), label=['Intuitive','Sensing'])

T = Thipe(X, y_T)
X_train, X_test, y_train, y_test = T.trainy(imbl=False)
X_train, X_test = T.reducy()
T.try_model(model=LogisticRegression(), label=['Feeling','Thinking'])

J = Thipe(X, y_J)
X_train, X_test, y_train, y_test = J.trainy(imbl=True)
X_train, X_test = J.reducy()
J.try_model(model=LogisticRegression(), label=['Perceiving','Judging'])

class NewBerd(object):

    def __init__(self):
        self.wordlist=[]
        self.sumdict = {'n_links':0,'n_image':0,'n_video':0,'emoticon_count':0,'mention_count':0,
                        'hashtag_count':0,'mbti_ref_count':0,'ennea_count':0,'bracket_count':0,'dots_count':0,'n_char':0,
                       'n_word':0,'n_action':0,'n_caps':0,'n_caps_char':0}
        self.avgdict={}
        self.taggedcollections=[]
        self.n_posts=0
        self.dff = pd.DataFrame()
        self.columns = []

    #Oh look what I just borrowed!

    def extractions_mod(self,line, string):
        """
        Input:
        x = A string of words
        string = regular expression that will match each word

        Output:
        lis = List of weblinks
        lis2 = List of 'words only' posts

        Taken from the preprocessing stage of the training set, except that now I am applying it to the user stage heheh.
        Instead of a list of sentences, we only have one string of words to deal with (each time).
        """
        lis=[]
        lin = line.split()
        sstring = re.compile(string, flags=re.M)
        lis_temp =[]
        for l in lin:
            if sstring.search(l):
                lis.append(sstring.search(l).group(0))
            else:
                lis_temp.append(l)
        lis_string = ' '.join(lis_temp)
        return lis, lis_string

    def onelvl_extract(self, x, string):
        """
        Input:
        x = A list of words
        string = regular expression in string form

        How the function works:

        Iterating through each word, if the word matches the regular expression, it will be added into the new list.
        The new list will hence be collecting all the re matched words that came from the input list with the chunk of sentences.
        This function returns the new list.
        """

        lis=[]
        sstring = re.compile(string, flags=re.M)
        for line in x.split():
            if sstring.search(line):
                lis.append(sstring.search(line).group(0))
        return lis

    def aggregate(self, words_only, tot_thing, string=''):
        tot_thing = tot_thing + len(self.onelvl_extract(words_only, string=string))
        n_thing = tot_thing * (50./len(self.wordlist))
        return tot_thing, n_thing

    def aggregate_web(self, weblink, tot_thing, string='.*'):
        tot_thing = tot_thing + len([y for y in weblink if re.match(string, y)])
        n_thing = tot_thing * (50./len(self.wordlist))
                                    #Special formula to match aggregate number with number of posts in training set
        return tot_thing, n_thing

    #Actual function 1
    def preprocess(self, input_string, web=True):
        self.wordlist.append(input_string)
        weblink, words_only = self.extractions_mod(line=input_string, string='https?://.*') #Weblink in list form, words_only in string form
        #Confirm again if we want to use web data
        if web:
            self.sumdict['n_links'], self.avgdict['n_links'] = self.aggregate_web(weblink, self.sumdict['n_links'])
            self.sumdict['n_image'], self.avgdict['n_image'] = self.aggregate_web(weblink, self.sumdict['n_image'],
                                                                   string='.*\.(jpg|png|jpeg|gif).*|.*img.*|.*image.*')
            self.sumdict['n_video'], self.avgdict['n_video'] = self.aggregate_web(weblink, self.sumdict['n_video'],
                                                                   string='https?://.*youtu.*|http.*vimeo.*')
            self.sumdict['n_otherlink'] = self.sumdict['n_links']-self.sumdict['n_image']-self.sumdict['n_video']
            self.avgdict['n_otherlink'] = self.sumdict['n_otherlink'] * (50./len(self.wordlist))
            self.sumdict['mention_count'], self.avgdict['mention_count'] = self.aggregate(words_only,
                                                                           tot_thing=self.sumdict['mention_count'], string='@\w*')
            self.sumdict['hashtag_count'], self.avgdict['hashtag_count'] = self.aggregate(words_only,
                                                                           tot_thing=self.sumdict['hashtag_count'],
                                                                           string='#\w[\w\d]*')
            self.sumdict['mbti_ref_count'], self.avgdict['mbti_ref_count'] = self.aggregate(words_only,
                                                                             tot_thing=self.sumdict['mbti_ref_count'],
                                                                             string='[eiEI][snSN][tfTF][jpJP]')
            self.sumdict['ennea_count'], self.avgdict['ennea_count'] = self.aggregate(words_only,
                                                                       tot_thing=self.sumdict['ennea_count'], string='\dw\d')
            self.sumdict['bracket_count'], self.avgdict['bracket_count'] = self.aggregate(words_only,
                                                                           tot_thing=self.sumdict['bracket_count'],
                                                                           string='\[.*?\]')
        else:
            #Get rid of the dictionary keys not used
            map(lambda x: self.sumdict.pop(x, None), ['n_video','n_links','n_image','mention_count','hashtag_count',
                                                      'mbti_ref_count','ennea_count','bracket_count'])


        self.sumdict['emoticon_count'], self.avgdict['emoticon_count'] = self.aggregate(words_only,
                                                                         tot_thing=self.sumdict['emoticon_count'], string=':\w*:')
        self.sumdict['dots_count'], self.avgdict['dots_count'] = self.aggregate(words_only,
                                                                tot_thing=self.sumdict['dots_count'], string='\.\.\.')
        self.sumdict['n_action'], self.avgdict['n_action'] = self.aggregate(words_only,
                                                             tot_thing=self.sumdict['n_action'], string='\*\w.*\*')
        #This one abit special
        capp = self.onelvl_extract(words_only, string=r'(?!([eiEI]?[snSN][tfTF][jpJP]|MBTI))[A-Z]{3,}')
        self.sumdict['n_caps'] += len(capp)
        self.avgdict['n_caps'] = self.sumdict['n_caps'] * (50./len(self.wordlist))
        self.sumdict['n_caps_char'] += np.sum([len(y) for y in capp])
        self.avgdict['n_caps_char'] = self.sumdict['n_caps_char'] * (50./len(self.wordlist))
        self.sumdict['n_char'] += len(words_only)
        self.avgdict['n_char'] = self.sumdict['n_char'] * (50./len(self.wordlist))
        self.sumdict['n_word'] += len(words_only.split())
        self.avgdict['n_word'] = self.sumdict['n_word'] * (50./len(self.wordlist))
        #No need sumdict
        self.avgdict['word_cap_ratio'] = float(self.sumdict['n_caps']) / self.sumdict['n_word']
        self.avgdict['char_cap_ratio'] = float(self.sumdict['n_caps_char']) / self.sumdict['n_char']
        self.avgdict['med_char'] = np.median([len(y) for y in self.wordlist])
        self.avgdict['med_word'] = np.median([len(y.split()) for y in self.wordlist])
        self.avgdict['word_char_ratio'] = self.avgdict['med_char'] / self.avgdict['med_word']

        #Save to dataframe
        self.dff = pd.DataFrame(self.avgdict, index=[1])

        #Create list and dictionary of POS tagging based on existing labels extracted from before
        convtag_dict={'ADJ':['JJ','JJR','JJS'], 'ADP':['EX','TO'], 'ADV':['RB','RBR','RBS','WRB'], 'CONJ':['CC','IN'],'DET':['DT','PDT','WDT'],
              'NOUN':['NN','NNS','NNP','NNPS'], 'NUM':['CD'],'PRT':['RP'],'PRON':['PRP','PRP$','WP','WP$'],
              'VERB':['MD','VB','VBD','VBG','VBN','VBP','VBZ'],'.':['#','$',"''",'(',')',',','.',':'],'X':['FW','LS','UH']}
        tagg = nltk.pos_tag(word_tokenize(input_string.decode('utf-8', errors='replace')))
        self.taggedcollections.append(tagg)
        lollie = ['#','$',"''",'``','(',')',',','.',':','CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB',
                  'RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']

        #Create columns to get the mean and std for each POS tagging for each row
        for col in lollie:
            self.dff['POS_'+col+'_mean'] = [np.mean([len([y for y in line if y[1] == col])for line in self.taggedcollections])]
            self.dff['POS_'+col+'_std'] = [np.std([len([y for y in line if y[1] == col])for line in self.taggedcollections])]
        for col in convtag_dict.keys():
            self.dff['BIGPOS_'+col+'_med'] = [np.median([len([y for y in line if y[1] in convtag_dict[col]])
                                                   for line in self.taggedcollections])]
            self.dff['BIGPOS_'+col+'_std']= [np.std([len([y for y in line if y[1] in convtag_dict[col]])
                                                   for line in self.taggedcollections])]

    def perform_magic(self, M):
        """
        This function takes in the instance for each typology class and comes out with the prediction
        Basically the most important function!
        """
        ok_df = pd.DataFrame()
        chunkie = [' '.join(self.wordlist)]
        for i,(tfidf, tsvd) in enumerate(zip(M.tfidf_list, M.tsvd_list)):
            wowwie = tsvd.transform(tfidf.transform(chunkie))
            da_df = pd.DataFrame(wowwie, columns=[str(i+1)+'_'+str(b) for b in np.arange(1,wowwie.shape[1]+1)])
            ok_df = pd.concat([ok_df,da_df], axis=1)

        #Set columns
        self.dff = self.dff[M.columns]
        column1 = self.dff.columns
        stded = M.ss.transform(self.dff)
        combinedf = pd.DataFrame(stded, columns=column1).join(ok_df)
        column2 = combinedf.columns
        combinedf = pd.DataFrame(M.mms.transform(combinedf), columns=column2)
        test = M.ch2.transform(combinedf)
        magic = M.sexy.predict(test)
        return magic


Yixuan = NewBerd()
entry = 'I dunno lah see how lor...'
Yixuan.preprocess(entry, web=False)
entry = 'Hey hey you you I know that you like me' #Just change this portion and run from here every time
Yixuan.preprocess(entry, web=False)

typerogy = lambda x, y: y[1] if x==[1] else y[0]
type1 = typerogy(Yixuan.perform_magic(E), ['Introvert','Extrovert'])
type2 = typerogy(Yixuan.perform_magic(S), ['iNtuitive','Sensing'])
type3 = typerogy(Yixuan.perform_magic(T), ['Feeling','Thinking'])
type4 = typerogy(Yixuan.perform_magic(J), ['Perceiving','Judging'])
print type1+' '+type2+' '+type3+' '+type4
