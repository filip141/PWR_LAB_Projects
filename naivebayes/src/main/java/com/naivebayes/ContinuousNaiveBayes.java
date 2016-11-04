package com.naivebayes;

import java.util.*;

import static java.lang.Math.exp;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

/**
 * Created by filip on 27.10.16.
 */
public class ContinuousNaiveBayes extends NaiveBayes {
    public ContinuousNaiveBayes(String dataPath, boolean classPosition) {
        super(dataPath, classPosition, false, 20, true, 20);
        this.trainingSet.normTrainingSet();
    }

    public ContinuousNaiveBayes(String dataPath){
        super(dataPath, false, false, 20, true, 20);
        this.trainingSet.normTrainingSet();
    }

    public void shuffleTrainingSet(){
        this.trainingSet.shuffle();
        this.trainingSet.normTrainingSet();
    }

    // Calculate probability from Gaussian
    public static Double calculateGaussianProb(Double x,Double mean,Double stdev){
        Double stdPower = 2*pow(stdev, 2);
        if(!stdPower.equals(0.0)){
            Double exponent = exp(-pow(x-mean, 2) / stdPower);
            return (1 / (sqrt(2*Math.PI) * stdev)) * exponent;
        }
        return 1.0;
    }

    public int predict(List<Double> prediction){
        double[] means;
        double[] stdevs;
        double tmpPrediction;
        int finalResult = 0;
        List <Observation> trainByID;
        List<Double> predictionNew = trainingSet.normalizeRecord(prediction);

        // Define predict classes and map
        Set<Double> classes = trainingSet.getClasses();
        Map<Double, Double> predictResults = new HashMap<Double, Double>();
        for(Double nbClass: classes){
            tmpPrediction = 1;
            // Define mean and std_dev
            trainByID  = TrainingSet.getTrainingDataByClass(nbClass.intValue(), trainingSet.getTrainingData());
            means = TrainingSet.mean(trainByID);
            stdevs = TrainingSet.stddev(trainByID);
            for(int i = 0; i < predictionNew.size(); i++){
                tmpPrediction *= calculateGaussianProb(predictionNew.get(i), means[i], stdevs[i]);
            }
            predictResults.put(nbClass, (trainByID.size() /
                    (1.0 * trainingSet.getTrainingData().size())) * tmpPrediction);
        }
        Double maxValueInMap = (Collections.max(predictResults.values()));
        for (Map.Entry<Double, Double> entry : predictResults.entrySet()) {
            if (entry.getValue().equals(maxValueInMap)) {
                finalResult = entry.getKey().intValue();
            }
        }
        return finalResult;
    }


}
