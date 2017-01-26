package com.knnproject;

import java.io.IOException;
import java.util.*;


public abstract class MachineLearningAlgorithm {

    protected TrainingSet trainingSet;

    public MachineLearningAlgorithm(String dataPath, boolean classPosition, boolean discreteValues, int bins,
                                    boolean equalFrequency, int efRec, boolean randomCv){
        try {
            this.trainingSet =new TrainingSet(dataPath, classPosition, discreteValues, bins, equalFrequency,
                    efRec, randomCv);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public abstract int predict(List<Double> prediction);

    public abstract void shuffleTrainingSet();

    public TrainingSet getTrainingSet(){
        return this.trainingSet;
    }

}
