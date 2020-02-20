"""
CNN Model for transfer learning
"""

class CNN:
    def __init__(self, 
                 total_classes=2,model_base='Xception',learning_rate=0.01,decay=1e-4,model_name='modelx',
                 input_shape=(256,256,3),iters=256,loss='binary_crossentropy',batch_size=32,freeze_base_model=False
                ):
        self.total_classes = total_classes
        self.model_base = model_base
        self.input_shape=input_shape
        self.model_name = model_name
        self.decay = decay
        self.lr = learning_rate
        self.model = None
        self.iters= iters
        self.loss = loss
        self.batch_size=batch_size
        self.freeze_base_model=freeze_base_model
    def get_model(self):
        def get_layer_name(ilayer_names,layer_type='dense'):
            ltns = [0]
            for _ in ilayer_names:
                if layer_type in _:
                    layer_type_n = int(_.split("_")[-1])
                    ltns.append(layer_type_n)
            ltn = layer_type+"_"+str(max(ltns)+1)
            ilayer_names.append(ltn)
        layer_names=[]
        from keras.applications import MobileNet
        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        from keras.applications.densenet import DenseNet121
        from keras.applications.xception import Xception
        from keras.layers import Dense,GlobalAveragePooling2D
        from keras.models import Model
        from keras.optimizers import Adam,Adadelta
        from keras.layers import Flatten
        from keras.models import Model
        from matplotlib.path import Path
        from keras.callbacks import ModelCheckpoint
        from keras.optimizers import Adadelta, Adam
        from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, BatchNormalization, Activation, Concatenate, Input,Dense
        from keras import backend as K
        K.clear_session()
        models = {'MobileNet':MobileNet,'InceptionResNetV2':InceptionResNetV2,'DenseNet121':DenseNet121,'Xception':Xception}
        base_model = models[self.model_base](include_top=False, weights='imagenet',input_shape=self.input_shape)
        if self.freeze_base_model==True:
            base_model.trainable=False
            for layer in base_model.layers:
                layer.trainable=False
        x=base_model.output
        x = Flatten()(x)
        x=Dense(1024,activation='relu',name=get_layer_name(layer_names,layer_type='dense'))(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
        x=Dense(1024,activation='relu',name=get_layer_name(layer_names,layer_type='dense'))(x) #dense layer 2
        x=Dense(512,activation='relu',name=get_layer_name(layer_names,layer_type='dense'))(x) #dense layer 3
        preds=Dense(self.total_classes,activation='softmax',name='output')(x) #final layer with softmax activation
        self.model=Model(inputs=base_model.input,outputs=preds,name=self.model_name)
        self.model.compile(optimizer=Adam(lr=self.lr, decay=self.decay),loss=self.loss)
        print("Model created. Type obj.model.summary() to get its architecture.")
    def FCL(self,previous_layer,activation='relu',batch_norm=True,n=128,scope='ChemNet',layer_name='FC'):
        x = Dense(units=n,name=self.get_name(layer_name,scope))(previous_layer)
        if batch_norm==True:
            x = BatchNormalization(name=self.get_name("batch_norm", scope))(x)
        x = Activation(activation, name=self.get_name(activation, scope))(x)
        return x
    def get_name(self,prefix, scope):
        counter = self.layer_names.get(prefix, 0)
        counter += 1
        name = scope + "_" + prefix + '_' + str(counter)
        self.layer_names[prefix] = counter
        return name
    def fit(self,ix,iy):
        self.get_model()
        self.train(ix,iy)
    def train(self,ix,iy):
        self.model.fit(ix,iy,epochs=self.iters,batch_size=self.batch_size)
    def model_predict(self,ix):
        return self.model.predict(ix)
    def predict_proba(self,ix):
        return self.model_predict(ix)
    def predict(self,ix):
        return self.model_predict(ix).argmax(1)
