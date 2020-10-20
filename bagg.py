from microtc.textmodel import TextModel
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from scipy.optimize import minimize, shgo, differential_evolution
from flair.embeddings import (CharacterEmbeddings, DocumentPoolEmbeddings, BytePairEmbeddings,TransformerWordEmbeddings,
                              DocumentRNNEmbeddings,BytePairEmbeddings,DocumentLSTMEmbeddings,FlairEmbeddings)
from flair.data import Sentence
from joblib import Parallel, delayed
#ml='models/malayalam_params.json'
#l=json.load(open(ml))
#lm={}
#mlf='data/malayalam_train.json'
#mx=pd.read_json(mlf, lines=True)
#X=mx.text.values
#Y=mx.klass.values
#le = LabelEncoder()
#le.fit(Y)
#y=le.transform(Y)
#Xt,Xv,yt,yv=train_test_split(X,y,test_size=0.2,random_state=33, stratify=y)

class bagginTextModels():
    def _train_model(self,model):
        xt=model.get('xt')
        print(f"Training model : {model['name']}, {xt.shape}")
        clf=LinearSVC()
        #clf=model['name']=='mtc' and LinearSVC() or RidgeClassifier()#RandomForestClassifier()
        if 'xv'in model.keys()  :
            xv=model.get('xv')
            model['clf']=clf.fit(xt,self.yt)#LinearSVC(max_iter=5000).fit(xt,self.yt)
            #model['clf']=RandomForestClassifier().fit(xt,self.yt)
            #if model['name']=='mtc':
            yp=model['clf'].decision_function(xv)
            #else:
            #    yp=model['clf'].predict_proba(xv)
            yp=Normalizer().fit_transform(yp)
            model['macroF1']=f1_score(self.yv,np.argmax(yp,axis=1),average='macro')
            model['weightedF1']=f1_score(self.yv,np.argmax(yp,axis=1),average='weighted')
            model['probas']=yp
            ### Fit model with all avaliable data
        else:
            model['clf']=clf.fit(xt,self.y)
        return model
        
    def __create_models(self):
        models=[]
        models_fit=[]
        #for _params in self.model_params:
        _params={}
        for k,v in self.params.items():
            if k.startswith('_'):
                continue
            _params[k]=v
        self.textModels=dict(mtc=TextModel(_params).fit(self.train),
                             #charEmb=DocumentPoolEmbeddings([CharacterEmbeddings()]),
                             #charLangEmb=DocumentPoolEmbeddings([CharacterEmbeddings(),BytePairEmbeddings(self.lang)]),
                             ##charMultiEmb=DocumentPoolEmbeddings([CharacterEmbeddings(),BytePairEmbeddings('multi')]),
                             langEmb=DocumentPoolEmbeddings([BytePairEmbeddings(self.lang)]),
                             charLangMultiEmb=DocumentPoolEmbeddings([CharacterEmbeddings(),BytePairEmbeddings(self.lang),
                                                                       BytePairEmbeddings('multi')]),
                             langMultiEmb=DocumentPoolEmbeddings([BytePairEmbeddings(self.lang),BytePairEmbeddings('multi')]),
                             bytePairEMB=DocumentPoolEmbeddings([BytePairEmbeddings('multi')]),
                             #flairEmbF=DocumentPoolEmbeddings([FlairEmbeddings('multi-forward')]),
                             #flairEmbB=DocumentPoolEmbeddings([FlairEmbeddings('multi-backward')]),
                             #bertEMB=DocumentPoolEmbeddings([TransformerWordEmbeddings('bert-base-uncased', layers='-1')])
        )
        for km,tmodel in self.textModels.items():
            models.append({'name':km})
            models_fit.append({'name':km})
            if km=='mtc':
                xt=tmodel.transform(self.train)
                xv=tmodel.transform(self.validation)
                X=tmodel.transform(self.data)
            else:
                sentences_train=[Sentence(txt) for txt in self.train]
                tmodel.embed(sentences_train)
                xt=np.array([e.get_embedding().cpu().detach().numpy() for e in sentences_train])
                sentences_val=[Sentence(txt) for txt in self.validation]
                tmodel.embed(sentences_val)
                xv=np.array([e.get_embedding().cpu().detach().numpy() for e in sentences_val])
                sentences=[Sentence(txt) for txt in self.data]
                tmodel.embed(sentences)
                X=np.array([e.get_embedding().cpu().detach().numpy() for e in sentences])
            models[-1]['xv']=xv
            models[-1]['xt']=xt
            models_fit[-1]['xt']=X
            #max_iter=5000
            #if km=='mtc': max_iter=1000
            #if km=='langMulti': max_iter=5000
            #self.models[-1]['clf']=LinearSVC(max_iter=max_iter).fit(xt,self.yt)
            #yp=self.models[-1]['clf'].decision_function(xv)
            #scaler=Normalizer().fit(yp)
            #self.models[-1]['macroF1']=f1_score(self.yv,np.argmax(scaler.transform(yp),axis=1),average='weighted')
            #self.models[-1]['weightedF1']=f1_score(self.yv,np.argmax(scaler.transform(yp),axis=1),average='weighted')
            #self.models[-1]['score']=f1_score(self.yv,np.argmax(yp,axis=1),average='weighted')
            #self.models[-1]['probas']=scaler.transform(yp)
            ### Fit model with all avaliable data
            #self.models_fit[-1]['clf']=LinearSVC(max_iter=max_iter).fit(X,self.y)
        print('Fitting Ensemble')
        #self.models  =  Parallel(n_jobs=5)(delayed(self._train_model)(md) for md in models)
        #self.models_fit = Parallel(n_jobs=5)(delayed(self._train_model)(md) for md in models_fit)
        self.models,self.models_fit=[],[]
        for md,mdf in zip(models, models_fit):
            self.models.append(self._train_model(md))#  =  [self._train_model(md) for md in models]
            self.models_fit.append(self._train_model(md))
            
    def __init__(self,data,labels,model_params, test_data=None, test_labels=None,test_size=0.2, lang='ml'):
        self.lang=lang
        self.labels=labels
        self.params=model_params
        self.LabelEncoder = LabelEncoder()
        self.LabelEncoder.fit(labels)
        self.y=self.LabelEncoder.transform(labels)
        if test_data is None:
            self.data=data
            self.train,self.validation,self.yt,self.yv=train_test_split(self.data,self.y,test_size=0.2,
                                                                        random_state=33, stratify=self.y)
        else:
            self.data=np.concatenate((data,test_data),axis=0)
            self.labels=np.concatenate((labels,test_labels))
            self.train,self.validation,self.yt,self.yv=data,test_data,self.y,self.LabelEncoder.transform(test_labels)
            self.y=self.LabelEncoder.transform(self.labels)
        self.__create_models()
        
    def dotF1(self,alphas):
        probas=alphas[0]*self.models[0]['probas']
        for alpha,model in zip(alphas[1:],self.models[1:]):
            probas=probas+alpha*model['probas']
        yp=np.argmax(probas,axis=1)
        #return -f1_score(self.yv,yp,average='weighted')
        return -f1_score(self.yv,yp,average='weighted')#*f1_score(self.yv,yp,average='macro')

    def one_constraint(self,alphas):
        return 1-alphas.sum()

    def optimize(self):
        cons=[{'type':'ineq','fun': self.one_constraint}]
        n_models=len(self.models)
        #alphas0=np.array([1/n_models for i in range(n_models)])
        scores=np.array([model['macroF1'] for model in self.models])
        alphas0=scores/scores.sum()
        #print("x0",alphas0)
        bnds=[(0.0,1.0) for alpha in alphas0]
        #sol=minimize(self.dotF1,alphas0,method='SLSQP',constraints=cons,bounds=bnds)
        sol=differential_evolution(self.dotF1,bnds)
        #sol=shgo(self.dotF1,constraints=cons,bounds=bnds)
        self.sol=sol
        self.weights=sol.x
        
    def predict(self,X):
        #sentences=[Sentence(txt) for txt in X]
        ft=True
        Yp=[]
        for w, model in zip(self.weights,self.models_fit):
            if model['name']=='mtc':
                x=self.textModels['mtc'].transform(X)
            else:
                sentences=[Sentence(txt) for txt in X]
                self.textModels[model['name']].embed(sentences)
                x=np.array([e.get_embedding().cpu().detach().numpy() for e in sentences])
            #if  model['name']=='mtc':
            yp=model['clf'].decision_function(x)
            #else:
            #    yp=model['clf'].predict_proba(x)
            if ft:
                Yp=w*Normalizer().fit_transform(yp)
                ft=False
            else:
                Yp=Yp+w*Normalizer().fit_transform(yp)
        return list(np.argmax(Yp,axis=1))

import pickle,gc
if __name__=='__main__':
    from bagg import bagginTextModels
    for lang,desc in [('ml','malayalam'),('ta','tamil')]:
    #for lang,desc in [('ta','tamil')]:
        gc.collect()
        tl=f'models/{desc}_params.json'
        tp=json.load(open(tl))
        tlf=f'data/{desc}_train.json'
        data=pd.read_json(tlf, lines=True)
        tX=data.text.values
        tY=data.klass.values
        tdata=pd.read_json(f'data/{desc}_dev.json', lines=True)
        xtt=[txt for txt in tdata.text.values]
        ytt=[txt for txt in tdata.klass.values]
        #bm=bagginTextModels(tX,tY,tp[0],xtt,ytt,lang=lang)
        bm=bagginTextModels(tX,tY,tp[0],lang=lang)
        bm.optimize()
        ypp=bm.predict(xtt)
        [print((model['name'], model['macroF1'], model['weightedF1'])) for model in bm.models]
        print('pred',precision_score(bm.LabelEncoder.transform(ytt),ypp, average='weighted'))
        print('rec',recall_score(bm.LabelEncoder.transform(ytt),ypp, average='weighted'))
        print('Wieghted',f1_score(bm.LabelEncoder.transform(ytt),ypp, average='weighted'))
        pickle.dump(bm,open(f'{desc}_model_dev_final.pk','wb'))
