package com.naivebayes;

import java.util.*;

import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

/**
 * Created by filip on 18.10.16.
 */
class Observation {
    public List<Double> attributes;
    public int label;

    public Observation(List<Double> attr,int label){
        this.attributes = attr;
        this.label = label;
    }

    public Observation clone(){
        List<Double> newList = new ArrayList<Double>();
        for(Double num: attributes){
            newList.add(num);
        }

        Observation o = new Observation(newList, this.label);

        return o;
    }
}

class TrainingSet {

    public final static double dataSetParts = 10.0;

    double[] meanVal;
    double[] std;
    private List<List<Observation>> crossValidationSets;
    private List<Observation> trainingSet;
    private List<Observation> testSet;
    private Set<Double> classes;
    private int testIndex;

    // Training set constructor
    public TrainingSet(List<List<Double>> fullSet, boolean classPosition){

        //Private Initialization
        crossValidationSets = new ArrayList<List<Observation>>();
        trainingSet = new ArrayList<Observation>();
        testSet = new ArrayList<Observation>();
        classes = new HashSet<Double>();

        // Local variables initialization
        List<Observation> fullObservationSet = new ArrayList<Observation>();
        Map<Double, List<Observation>> classRecords = new HashMap<Double, List<Observation>>();
        Map<Double, Integer> numClassRec = new HashMap<Double, Integer>();
        List<Observation> tmpObservationList;
        List<Double> obsAttributes;
        Double classObservation;
        int obsNumber;

        // Find Data Set classes
        for(List<Double> singleObservation: fullSet){
            if(classPosition){
                obsAttributes = singleObservation.subList(1, singleObservation.size());
                classObservation = singleObservation.get(0);
            }
            else{
                obsAttributes = singleObservation.subList(0, singleObservation.size() - 2);
                classObservation = singleObservation.get(singleObservation.size() - 1);
            }
            classes.add(classObservation);
            fullObservationSet.add(new Observation(obsAttributes, classObservation.intValue()));
        }

        // Assign Observation
        for(Double signleClass: classes){
            List<Observation> tmpRecList = TrainingSet.getTrainingDataByClass(signleClass.intValue(),
                    fullObservationSet);
            classRecords.put(signleClass, tmpRecList);
            numClassRec.put(signleClass, tmpRecList.size());
        }

        for(int i = 0; i < TrainingSet.dataSetParts; i++){
            tmpObservationList = new ArrayList<Observation>();
            for(Double signleClass: classes) {
                if(i == TrainingSet.dataSetParts - 1){
                    tmpObservationList.addAll(classRecords.get(signleClass));
                }
                else{
                    obsNumber = (int) (numClassRec.get(signleClass) / TrainingSet.dataSetParts);
                    for(int j = 0; j < obsNumber; j++){
                        tmpObservationList.add(classRecords.get(signleClass).get(0));
                        classRecords.get(signleClass).remove(0);
                    }
                }
            }
            crossValidationSets.add(tmpObservationList);
        }

        // Select First Training Set
        testIndex = 0;
        List<Observation> copiedList = new ArrayList<Observation>();
        for(Observation obs: crossValidationSets.get(testIndex)){
            copiedList.add(obs.clone());
        }

        testSet.addAll(copiedList);
        for(List<Observation> cvSet: crossValidationSets.subList(1, crossValidationSets.size())){
            List<Observation> copiedCvSet = new ArrayList<Observation>();
            for(Observation obs: cvSet){
                copiedCvSet.add(obs.clone());
            }
            trainingSet.addAll(copiedCvSet);
        }


        // Calculate mean for existing dataset
        meanVal = mean(trainingSet);
        std = stddev(trainingSet);
    }

    public void shuffle(){
        // Clear sets
        testSet = new ArrayList<Observation>();
        trainingSet = new ArrayList<Observation>();

        testIndex += 1;
        List<Observation> copiedList = new ArrayList<Observation>();
        for(Observation obs: crossValidationSets.get(testIndex)){
            copiedList.add(obs.clone());
        }

        testSet.addAll(copiedList);
        for(int i = 0; i < crossValidationSets.size(); i++){
            if(i != testIndex){
                List<Observation> copiedCvSet = new ArrayList<Observation>();
                for(Observation obs: crossValidationSets.get(i)){
                    copiedCvSet.add(obs.clone());
                }
                trainingSet.addAll(copiedCvSet);
            }
        }

        // Calculate new mean and stddev
        meanVal = mean(trainingSet);
        std = stddev(trainingSet);
    }

    public static double[] sum(List<Observation> dataSet){
        int arrayLen = dataSet.get(0).attributes.size();
        double[] result = new double[arrayLen];

        for(Observation singl: dataSet){
            for(int i=0; i < singl.attributes.size(); i++){
                result[i] += singl.attributes.get(i);
            }
        }
        return result;
    }

    public static double[] mean(List<Observation> dataSet){
        double[] attrSum = sum(dataSet);
        int arrayLen = dataSet.get(0).attributes.size();
        double[] result = new double[arrayLen];

        for(int i=0; i < arrayLen; i++){
            result[i] = attrSum[i] / dataSet.size();
        }

        return result;
    }

    public static double[] stddev(List<Observation> dataSet){
        int arrayLen = dataSet.get(0).attributes.size();
        double[] result = new double[arrayLen];
        double[] meanArray = mean(dataSet);

        for(Observation singl: dataSet){
            for(int i=0; i < singl.attributes.size(); i++){
                result[i] += pow((singl.attributes.get(i) - meanArray[i]), 2);
            }
        }

        for(int i=0; i < arrayLen; i++){
            result[i] = sqrt(result[i] / dataSet.size());
        }

        return result;
    }

    public static List<Observation> getTrainingDataByClass(int label, List<Observation> dataSet){
        List<Observation> newTrainingData = new ArrayList<Observation>();
        for(Observation obs: dataSet){
            if (obs.label == label){
                newTrainingData.add(obs);
            }
        }
        return newTrainingData;
    }

    public List<Observation> normalizeObsRec(List<Observation> obsInput){
        double tmpFeature;
        List<Observation> newObsRec = new ArrayList<Observation>();

        for(Observation singl: obsInput){
            List<Double> newAttrs = new ArrayList<Double>();
            for(int i=0; i < singl.attributes.size(); i++){
                tmpFeature = (singl.attributes.get(i) - meanVal[i]) / std[i];
                newAttrs.add(tmpFeature);
            }
            singl.attributes = newAttrs;
            newObsRec.add(singl);
        }

        return newObsRec;
    }

    public List<Double> normalizeRecord(List<Double> rawInput){
        double tmpFeature;
        List<Double> newAttrs = new ArrayList<Double>();

        for(int i=0; i < rawInput.size(); i++){
            tmpFeature = (rawInput.get(i) - meanVal[i]) / std[i];
            newAttrs.add(tmpFeature);
        }

        return newAttrs;
    }

    public void normTrainingSet(){

        double tmpFeature;
        List<Observation> newTrainingSet = new ArrayList<Observation>();

        for(Observation singl: trainingSet){
            List<Double> newAttrs = new ArrayList<Double>();
            for(int i=0; i < singl.attributes.size(); i++){
                tmpFeature = (singl.attributes.get(i) - meanVal[i]) / std[i];
                newAttrs.add(tmpFeature);
            }
            singl.attributes = newAttrs;
            newTrainingSet.add(singl);
        }

        this.trainingSet = newTrainingSet;

    }

    public List<Observation> getTrainingData(){
        return trainingSet;
    }

    public List<Observation> getTestData(){
        return testSet;
    }

    public Set<Double> getClasses(){
        return classes;
    }
}
