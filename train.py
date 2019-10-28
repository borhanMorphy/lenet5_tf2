from dataset import get_dataset
from model import LeNet5
import tensorflow as tf

if __name__ == '__main__':
    # create model
    model = LeNet5()

    # initialize model train configs
    batch_size = 100
    epochs = 20
    data_set_size = 42000
    validation_fraction = 0.1
    
    # compile the model
    model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
    
    # get dataset
    train_ds,val_ds = get_dataset("train",data_set_size=data_set_size,split_val=validation_fraction)
    
    # load all data to the gpu memory for faster operations since data is small
    tf.print("Loading dataset to the gpu memory... This may take a while") 
    train_ds = list(train_ds)
    x_train,y_train = zip(*train_ds)
    x_train = tf.stack(x_train,axis=0)
    y_train = tf.stack(y_train,axis=0)

    val_ds = list(val_ds)
    x_val,y_val = zip(*val_ds)
    x_val = tf.stack(x_val,axis=0)
    y_val = tf.stack(y_val,axis=0)
    tf.print("Dataset is loaded.")
    # train the model
    model.fit(x=x_train,y=y_train,epochs=epochs,steps_per_epoch=data_set_size//batch_size,batch_size=batch_size,shuffle=True,
        verbose=1,validation_data=(x_val,y_val))
    
    # save model for deployment
    tf.saved_model.save(model, "models_for_serving/lenet5/1/")
    