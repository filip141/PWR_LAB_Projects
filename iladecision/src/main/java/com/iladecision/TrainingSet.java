package com.iladecision;

import java.io.IOException;
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

    private double bins;
    private int efRecords;
    private double[] meanVal;
    private double[] std;
    private boolean discreteValues;
    private List<List<double[]>> discretizationRegions;
    private List<List<Observation>> crossValidationSets;
    private List<Observation> trainingSet;
    private List<Observation> testSet;
    private Set<Double> classes;
    private int testIndex;

    // Training set constructor
    public TrainingSet(String dataPath, boolean classPosition, boolean discreteValues, int bins,
                       boolean equalFrequency, int efRecords, boolean randomCv) throws IOException {

        // Full set initialization
        List<List<Double>> fullSet;

        //Private Initialization
        crossValidationSets = new ArrayList<List<Observation>>();
        trainingSet = new ArrayList<Observation>();
        testSet = new ArrayList<Observation>();
        classes = new HashSet<Double>();
        this.discreteValues = discreteValues;

        //Get data from file
        DataFile dt = new DataFile(dataPath);
        fullSet = dt.getDataset();

        //Local variables
        List<Observation> fullObservationSet;

        // Generate Observation List
        fullObservationSet = generateObservationList(fullSet, classPosition);

        //Only for discrete values
        if(discreteValues){
            this.bins = bins;
            this.efRecords = efRecords;
            discretizationRegions = getDiscretizationRegions(fullObservationSet, equalFrequency);
            fullObservationSet = TrainingSetDiscretization(fullObservationSet);
        }

        if(randomCv){
            // Prepare CrossValidation Set
            prepareRandomCrossValidationSets(fullObservationSet);
        }
        else{
            // Prepare CrossValidation Set
            prepareCrossValidationSets(fullObservationSet);
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
//        trainingSet = fullObservationSet;//TODO

        // Not for discrete values
        if(!discreteValues){
            // Calculate mean for existing dataset
            meanVal = mean(trainingSet);
            std = stddev(trainingSet);
        }

    }
    public Observation discretize(Observation signleObservation){

        double feature;
        double regionMean;
        List<double[]> featureRegions;
        int featuresNumber = signleObservation.attributes.size();

        // Iterate over features
        for(int i = 0; i < featuresNumber; i++){
            // Get discretization regions
            feature = signleObservation.attributes.get(i);
            featureRegions = discretizationRegions.get(i);
            for (double[] region : featureRegions) {
                regionMean = (region[0] + region[1]) / 2;
                regionMean *= 100;
                regionMean = (double)((int) regionMean);
                regionMean /= 100;
                if(feature > region[0] && feature <= region[1]){
                    signleObservation.attributes.set(i, regionMean);
                }
            }
        }

        return signleObservation;
    }

    public List<Observation> TrainingSetDiscretization(List<Observation> fullObservationSet){

        for(int i = 0; i < fullObservationSet.size(); i++){
            Observation obs = discretize(fullObservationSet.get(i));
            fullObservationSet.set(i, obs);
        }

        return fullObservationSet;
    }

    public List<List<double[]>> getDiscretizationRegionsEqualWidth(List<Observation> fullObservationSet){

        Double maxFeature;
        Double minFeature;
        Double interval;
        double lowBoundary;
        double[] intervalBounds;
        List<double[]> featureInter;
        List<List<double[]>> intervals = new ArrayList<List<double[]>>();
        List<Double> featureList;
        int featuresNumber = fullObservationSet.get(0).attributes.size();

        // Get feature list
        for(int i = 0; i < featuresNumber; i++){

            featureInter = new ArrayList<double[]>();
            featureList = new ArrayList<Double>();

            for(Observation feature: fullObservationSet){
                featureList.add(feature.attributes.get(i));
            }

            maxFeature = Collections.max(featureList);
            minFeature = Collections.min(featureList);
            interval = (maxFeature - minFeature) / (1.0 * bins);

            if(interval.equals(0.0)){
                intervalBounds = new double[2];
                intervalBounds[0] = 0;
                intervalBounds[1] = 0;
                featureInter.add(intervalBounds);
                intervals.add(featureInter);
                continue;
            }

            lowBoundary = minFeature;

            while(lowBoundary <= maxFeature){
                intervalBounds = new double[2];
                intervalBounds[0] = lowBoundary;
                intervalBounds[1] = lowBoundary + interval;
                featureInter.add(intervalBounds);
                lowBoundary += interval;
            }
            intervals.add(featureInter);
        }
        return intervals;
    }


    public List<List<double[]>> getDiscretizationRegionsEqualFrequency(List<Observation> fullObservationSet){

        int counter;
        double[] intervalBounds;
        List<double[]> featureInter;
        List<List<double[]>> intervals = new ArrayList<List<double[]>>();
        List<Double> featureList;
        int featuresNumber = fullObservationSet.get(0).attributes.size();

        // Get feature list
        for(int i = 0; i < featuresNumber; i++){

            featureInter = new ArrayList<double[]>();
            featureList = new ArrayList<Double>();

            for(Observation feature: fullObservationSet){
                featureList.add(feature.attributes.get(i));
            }

            Collections.sort(featureList);

            counter = 0;
            while(counter < featureList.size() - efRecords){
                intervalBounds = new double[2];
                intervalBounds[0] = featureList.get(counter);
                intervalBounds[1] = featureList.get(counter + efRecords);
                counter += efRecords;
                featureInter.add(intervalBounds);
            }
            intervalBounds = featureInter.get(featureInter.size() - 1);
            intervalBounds[1] = featureList.get(featureList.size() - 1);
            featureInter.set(featureInter.size() - 1, intervalBounds);
            intervals.add(featureInter);
        }
        return intervals;
    }

    public List<List<double[]>> getDiscretizationRegions(List<Observation> fullObservationSet, boolean equalFrequency){

        List<List<double[]>> regions;

        if(equalFrequency){
            regions = getDiscretizationRegionsEqualFrequency(fullObservationSet);
        }
        else{
            regions = getDiscretizationRegionsEqualWidth(fullObservationSet);
        }

        return regions;
    }

    public List<Observation> generateObservationList(List<List<Double>> parsedRecords, boolean classPos){

        // Local variables initialization
        List<Observation> fullObservationSet = new ArrayList<Observation>();
        List<Double> obsAttributes;
        Double classObservation;

        // Find Data Set classes and create fullObservationSet
        for(List<Double> singleObservation: parsedRecords){
            if(classPos){
                obsAttributes = singleObservation.subList(1, singleObservation.size());
                classObservation = singleObservation.get(0);
            }
            else{
                obsAttributes = singleObservation.subList(0, singleObservation.size() - 1);
                classObservation = singleObservation.get(singleObservation.size() - 1);
            }
            classes.add(classObservation);
            fullObservationSet.add(new Observation(obsAttributes, classObservation.intValue()));
        }

        return fullObservationSet;
    }

    public void prepareRandomCrossValidationSets(List<Observation> fullObservationSet){

        int indexNumber;
        Random rand = new Random();
        List<Observation> tmpObservationList;

        // Prepare cross validation sets
        for(int i = 0; i < TrainingSet.dataSetParts; i++){
            tmpObservationList = new ArrayList<Observation>();
            crossValidationSets.add(tmpObservationList);
        }

        int i = 0;
        // Prepare cross validation sets
        while(!fullObservationSet.isEmpty()){
            tmpObservationList = crossValidationSets.get(i);
            indexNumber = rand.nextInt(fullObservationSet.size());
            tmpObservationList.add(fullObservationSet.get(indexNumber));
            crossValidationSets.set(i, tmpObservationList);
            fullObservationSet.remove(indexNumber);
            i = (i + 1) % (int) TrainingSet.dataSetParts;
        }
    }

    public void prepareCrossValidationSets(List<Observation> fullObservationSet){

        int obsNumber;
        Map<Double, List<Observation>> classRecords = new HashMap<Double, List<Observation>>();
        Map<Double, Integer> numClassRec = new HashMap<Double, Integer>();
        List<Observation> tmpObservationList;

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

    }

    public void shuffle(){
        // Clear sets
        testSet = new ArrayList<Observation>();
        trainingSet = new ArrayList<Observation>();

        testIndex = (int) ((testIndex + 1) % dataSetParts);
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

        if(!discreteValues){
            // Calculate new mean and stddev
            meanVal = mean(trainingSet);
            std = stddev(trainingSet);
        }
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
            result[i] = sqrt((dataSet.size() / (dataSet.size() - 1)) * (result[i] / dataSet.size()));
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
            if(std[i] != 0){
                tmpFeature = (rawInput.get(i) - meanVal[i]) / std[i];
            }
            else{
                tmpFeature = 0;
            }
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
                // Prevent Nans
                if(std[i] != 0) {
                    tmpFeature = (singl.attributes.get(i) - meanVal[i]) / std[i];
                }
                else{
                    tmpFeature = 0;
                }
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
