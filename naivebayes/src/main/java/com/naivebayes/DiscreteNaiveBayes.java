package com.naivebayes;

import java.io.IOException;
import java.util.*;

import static java.lang.Math.exp;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

/**
 * Created by filip on 27.10.16.
 */
package com.naivebayes;

/**
 * Created by filip on 19.10.16.
 */
public class DiscreteNaiveBayes extends NaiveBayes {


    public DiscreteNaiveBayes(String dataPath, boolean classPosition, boolean discretValues) {
        super(dataPath, classPosition, discretValues);
    }

    public int predict(List<Double> prediction){

        Observation obs;
        int finalResult = 0;
        List<Observation> trainingByID;
        obs = new Observation(prediction, 0);
        obs = trainingSet.discretize(obs);

        int counterAll;
        int counterStrike;
        double tmpPrediction;
        Set<Double> classes = trainingSet.getClasses();
        Map<Double, Double> predictResults = new HashMap<Double, Double>();
        for(Double nbClass: classes){
            tmpPrediction = 1.0;
            trainingByID = TrainingSet.getTrainingDataByClass(nbClass.intValue(), trainingSet.getTrainingData());
            counterAll = trainingByID.size();
            for(int i = 0; i < obs.attributes.size(); i++){
                counterStrike = 1;
                for (Observation trainObs: trainingByID){
                    if(trainObs.attributes.get(i).equals(obs.attributes.get(i))){
                        counterStrike += 1;
                    }
                }
                tmpPrediction *= counterStrike / (1.0 * counterAll);
            }
            predictResults.put(nbClass, tmpPrediction);
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
