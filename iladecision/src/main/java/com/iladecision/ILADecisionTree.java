package com.iladecision;

import java.io.IOException;
import java.util.*;

/**
 * Created by filip on 15.11.16.
 */
public class ILADecisionTree extends DecisionTree {

    private TrainingSet trainingSet;
    private List<PatternPair> rules;
    Map<Integer, List<Observation>> subTableMap;

    public ILADecisionTree(String dataPath, boolean classPosition, int bins, boolean equalFrequency,
                           int efRec, boolean randomCv){
        super(dataPath, classPosition, bins, equalFrequency, efRec, randomCv);
        try {
            this.trainingSet =new TrainingSet(dataPath, classPosition, true, bins, equalFrequency,
                    efRec, randomCv);
            divideSubTables();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void shuffleTrainingSet() {
        this.trainingSet.shuffle();
    }

    public void divideSubTables(){
        this.subTableMap = new HashMap<Integer, List<Observation>>();

        for(Double subTableClass: trainingSet.getClasses()){
            subTableMap.put(subTableClass.intValue(), TrainingSet.getTrainingDataByClass(subTableClass.intValue(),
                    trainingSet.getTrainingData()));
        }
    }

    public List<HashSet<Integer>> constructNCombination(int maxSize){
        List<HashSet<Integer>> combSet = new ArrayList<HashSet<Integer>>();
        List<List<Double>> perArray = new ArrayList<List<Double>>();
        List<Double> featureIndexList = new ArrayList<Double>();
        int numberOfFeatures = trainingSet.getTrainingData().get(0).attributes.size();
        // Generate List
        for(int i=0; i < numberOfFeatures; i++){
            featureIndexList.add((double)i);
        }

        for(int j=0; j < maxSize; j++){
            perArray.add(featureIndexList);
        }
        // Construct desired format
        List<LinkedHashSet<Double>> tmpCombHashSet = itemsPermutation(perArray, perArray.size() - 1);
        for(LinkedHashSet<Double> insSet: tmpCombHashSet){
            HashSet<Integer> tmpSet = new HashSet<Integer>();
            for(Double val: insSet){
                tmpSet.add(val.intValue());
            }
            if(tmpSet.size() == maxSize){
                combSet.add(tmpSet);
            }
        }
        return combSet;
    }

    public List<Observation> copyTable(List<Observation> tableToCopy){
        List<Observation> newList = new ArrayList<Observation>();

        for(Observation obs: tableToCopy){
            newList.add(obs.clone());
        }
        return newList;
    }

    public boolean checkValueInOtherTables(HashSet<Double> permSet, HashSet<Integer> permutation, int actClass){
        boolean cond;
        List<Observation> tmpSubList = trainingSet.getTrainingData();
        for (Observation obs : tmpSubList) {
            cond = true;
            List<Integer> listPattern = new ArrayList<Integer>(permutation);
            List<Double> listElem = new ArrayList<Double>(permSet);
            for (int i = 0; i < listPattern.size(); i++) {
                cond = cond && (obs.attributes.get(listPattern.get(i)).equals(listElem.get(i)));
            }

            if(cond) {
                if(obs.label != actClass){
                    return false;
                }
            }
        }
        // Item could not be found in other classes
        return true;
    }

    public boolean isInLocalTable(HashSet<Double> permSet, HashSet<Integer> permutation, List<Observation> localTable){
        boolean cond;
        for (Observation obs : localTable) {
            cond = true;
            List<Integer> listPattern = new ArrayList<Integer>(permutation);
            List<Double> listElem = new ArrayList<Double>(permSet);
            for (int i = 0; i < listPattern.size(); i++) {
                cond = cond && (obs.attributes.get(listPattern.get(i)).equals(listElem.get(i)));
            }

            if(cond) {
                return true;
            }
        }
        // Item not exist
        return false;
    }


    public List<Double> getColumn(int colIdx, List<Observation> actualTable){
        List<Double> tableColumn = new ArrayList<Double>();
        for(Observation obs: actualTable){
            tableColumn.add(obs.attributes.get(colIdx));
        }
        return tableColumn;
    }

    public List<LinkedHashSet<Double>> itemsPermutation(List<List<Double>> data,int step){
        List<Double> tmpList = data.get(step);
        List<LinkedHashSet<Double>> hsList = new ArrayList<LinkedHashSet<Double>>();
        if(step == 0){
            for(Double attr: tmpList){
                LinkedHashSet<Double> hsAttr = new LinkedHashSet<Double>();
                hsAttr.add(attr);
                hsList.add(hsAttr);
            }
            return hsList;
        }
        else{
            for(Double attr: tmpList){
                List<LinkedHashSet<Double>> hsListOld = itemsPermutation(data, step - 1);
                for(LinkedHashSet<Double> ins: hsListOld){
                    ins.add(attr);
                    hsList.add(ins);
                }
            }
            return hsList;
        }
    }


    public Pair findMostCommon(HashSet<LinkedHashSet<Double>> hsPerm){
        int occur;
        int occurInd;
        LinkedHashSet<Double> mc;
        List<Integer> occurences = new ArrayList<Integer>();
        List<LinkedHashSet<Double>> tmpHsParam = new ArrayList<LinkedHashSet<Double>>(hsPerm);
        for(LinkedHashSet<Double> setHs: tmpHsParam){
            occurences.add(Collections.frequency(tmpHsParam, setHs));
        }
        occurInd = occurences.indexOf(Collections.max(occurences));
        occur = occurences.get(occurences.indexOf(Collections.max(occurences)));
        mc = tmpHsParam.get(occurInd);
        return new Pair(mc, occur);
    }


    public void train(){
        int perm;
        int attributesNumber;
        PatternPair maxPair;
        rules = new ArrayList<PatternPair>();
        attributesNumber = trainingSet.getTrainingData().get(0).attributes.size();

        // Every Permutation
        for(Integer extClass: subTableMap.keySet()){
            perm = 1;
            List<Observation> localTable = copyTable(subTableMap.get(extClass));
            while(localTable.size() > 0){
                if(perm > attributesNumber){
                    break;
                }
                List<HashSet<Integer>> localPermutation = constructNCombination(perm);
                List<PatternPair> maxCounts = new ArrayList<PatternPair>();
                for(HashSet<Integer> singlePermutation: localPermutation){
                    HashSet<LinkedHashSet<Double>> hsPerm;

                    // Prepare are for counting permutation
                    List<List<Double>> perArray = new ArrayList<List<Double>>();
                    for(Integer idx: singlePermutation){
                        List<Double> singleCol = getColumn(idx, localTable);
                        perArray.add(singleCol);
                    }
                    List<LinkedHashSet<Double>> tmpHashSet = itemsPermutation(perArray, perArray.size() - 1);
                    hsPerm = new HashSet<LinkedHashSet<Double>>(tmpHashSet);
                    // Remove items from other lists
                    for(Iterator<LinkedHashSet<Double>> iter = hsPerm.iterator(); iter.hasNext();) {
                        LinkedHashSet<Double> hs = iter.next();
                        if(hs.size() != perm){
                            iter.remove();
                            continue;
                        }
                        if(!isInLocalTable(hs, singlePermutation, localTable)) {
                            iter.remove();
                            continue;
                        }
                        if(!checkValueInOtherTables(hs, singlePermutation, extClass)) {
                            iter.remove();
                        }
                    }
                    if(hsPerm.size() > 0){
                        // Find most common element
                        Pair mostCommon = findMostCommon(hsPerm);
                        maxCounts.add(new PatternPair(mostCommon, singlePermutation, extClass.doubleValue()));
                    }
                }
                if(maxCounts.size() == 0){
                    perm += 1;
                    continue;
                }

                // Find Max
                maxPair = maxCounts.get(0);
                for(PatternPair pair: maxCounts){
                    if(pair.occurences > maxPair.occurences){
                        maxPair = pair;
                    }
                }

                // Remove classified element
                boolean cond;
                for(Iterator<Observation> iter = localTable.iterator(); iter.hasNext();) {
                    cond = true;
                    Observation obs = iter.next();
                    List<Integer> listPattern = new ArrayList<Integer>(maxPair.pattern);
                    List<Double> listElem = new ArrayList<Double>(maxPair.element);
                    for(int i = 0; i < listPattern.size(); i++){
                        cond = cond && (obs.attributes.get(listPattern.get(i)).equals(listElem.get(i)));
                    }
                    // If observation found remove
                    if(cond){
                        iter.remove();
                    }
                }
                rules.add(maxPair);
            }
        }
    }


    public int predict(List<Double> prediction) {
        Random rand = new Random();
        Observation value = new Observation(prediction, 0);
        value = trainingSet.discretize(value);
        Set<Double> classesSet = trainingSet.getClasses();
        List<Double> classesList = new ArrayList<Double>(classesSet);
        int randidx = rand.nextInt(classesSet.size());

        boolean cond;
        for(PatternPair patCond: rules){
            cond = true;
            List<Integer> listPattern = new ArrayList<Integer>(patCond.pattern);
            List<Double> listElem = new ArrayList<Double>(patCond.element);
            for(int i = 0; i < listPattern.size(); i++){
                cond = cond && (value.attributes.get(listPattern.get(i)).equals(listElem.get(i)));
            }
            // If observation found remove
            if(cond){
                return (int) patCond.result;
            }
        }
        return classesList.get(randidx).intValue();
    }
}
