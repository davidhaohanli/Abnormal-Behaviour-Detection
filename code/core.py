from weight_matrix import *
from Feature_extraction import *
from Classifiers import *
import scipy.io
import cv2
import matplotlib.pyplot as plt


def uvPlot(u,v,labels):
    plt.figure()
    for ind,label in enumerate(labels):
        if label:
            plt.scatter(u[ind],v[ind],c='red')
        else:
            plt.scatter(u[ind], v[ind], c='blue')
    plt.ylabel('v')
    plt.xlabel('u')
    plt.title('u v plot')
    plt.legend()
    plt.show()

def load_data():
    u_seq_abnormal = scipy.io.loadmat('../ref_data/u_seq_abnormal.mat')['u_seq_abnormal']
    v_seq_abnormal = scipy.io.loadmat('../ref_data/v_seq_abnormal.mat')['v_seq_abnormal']

    fg_imgs = ['../ref_data/fg_pics/' + str(i + 1) + '.bmp' for i in range(200)]
    original_imgs = ['../ref_data/original_pics/' + str(i + 1).zfill(3) + '.tif' for i in range(200)]
    abnormal_fg_imgs = ['../ref_data/ab_fg_pics/' + str(i + 1) + '.bmp' for i in range(200)]

    return u_seq_abnormal,v_seq_abnormal,fg_imgs,original_imgs,abnormal_fg_imgs

def plot(realPos,labels,img,timerSet=True):
    #TODO LABEL TRUE POS ONLY
    for i, item in enumerate(realPos):
        if labels[i]:
            cv2.rectangle(img, (int(item[3]), int(item[1])), (int(item[2]), int(item[0])), (0, 0, 255))
        # cv2.putText(img, str(i), (int(item[3]), int(item[1]) - 5), font, 0.4, (255, 255, 0), 1)
    cv2.imshow('img', img)
    #TODO TIMERSET
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def main ():

    u_data,v_data,fg_imgs,original_imgs,abnormal_fg_imgs=load_data()
    weight = Weight_matrix().get_weight_matrix()

    thisFeatureExtractor = Feature_extractor(original_imgs,fg_imgs,abnormal_fg_imgs,u_data,v_data,weight)

    #TODO PARAM SELECTION
    train_data,train_labels = thisFeatureExtractor.get_features_and_labels(100,110)
    uvPlot(train_data[:,0],train_data[:,1],train_labels)


    classifiers = Classifiers(train_data,train_labels)

    #TODO PARAM SELECTION
    test_data, test_labels = thisFeatureExtractor.get_features_and_labels(120,150)

    for name,model in classifiers.models.items():
        for ind,original_img in enumerate(original_imgs[80:-1]):
            #TODO -80
            ind=ind+80
            pos,thisImg,_=thisFeatureExtractor.getPosition(fg_imgs,ind)
            #print(pos.shape)
            features,_=thisFeatureExtractor.get_features_and_labels(ind,ind+1,False)
            #print(features.shape)
            #print(features)

            #TODO delete
            labels=classifiers.models[name].predict(features)
            #uvPlot(features[:,0],features[:,1],labels)
            print('ind: ',ind,'abnormal?: ',labels.max())

            plot(pos,labels,thisImg)
        #print (labelsum)

        classifiers.prediction_metrics(test_data,test_labels,name)



if __name__ == '__main__':
    main()