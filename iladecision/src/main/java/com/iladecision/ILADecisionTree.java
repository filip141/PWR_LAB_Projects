package com.iladecision;

import java.io.IOException;
import java.util.*;

/**
 * Created by filip on 15.11.16.
 */

class PatternPair{
    public LinkedHashSet<Double> element;
    public HashSet<Integer> pattern;
    public int occurences;
    public double result;
    PatternPair(Pair pair, HashSet<Integer> pattern, Double result){
        this.element = pair.element;
        this.occurences = pair.occurences;
        this.pattern = pattern;
        this.result = result;
    }
}


class Pair{
    public LinkedHashSet<Double> element;
    public int occurences;
    Pair(LinkedHashSet<Double> element, int occurences){
        this.element = element;
        this.occurences = occurences;
    }
}


public class ILADecisionTree{

    private TrainingSet trainingSet;
    private List<PatternPair> rules;
    private Map<Integer, List<HashSet<Integer>>> featureCombination;
    private Map<HashSet<Integer>, Map<LinkedHashSet<Double>, List<Integer>>> subTablePresent;
    Map<Integer, List<Observation>> subTableMap;

    public ILADecisionTree(String dataPath, boolean classPosition, int bins, boolean equalFrequency,
                           int efRec, boolean randomCv){
        try {
            this.trainingSet =new TrainingSet(dataPath, classPosition, true, bins, equalFrequency,
                    efRec, randomCv);
            this.featureCombination = constructCombinations();
            divideSubTables();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void divideSubTables(){
        this.subTableMap = new HashMap<Integer, List<Observation>>();

        for(Double subTableClass: trainingSet.getClasses()){
            subTableMap.put(subTableClass.intValue(), TrainingSet.getTrainingDataByClass(subTableClass.intValue(),
                    trainingSet.getTrainingData()));
        }
    }

    public List<HashSet<Integer>> getAllCombination(List<Integer> objectList){
        int combinationSize;
        HashSet<Integer> uniqueObjects = new HashSet<Integer>(objectList);
        List<HashSet<Integer>> combinations = new ArrayList<HashSet<Integer>>();

        for(Integer obj: uniqueObjects){
            combinationSize = combinations.size();
            for (int i = 0; i < combinationSize; i++) {
                HashSet<Integer> insideSet = new HashSet<Integer>(combinations.get(i));
                insideSet.add(obj);
                combinations.add(insideSet);
            }
            HashSet<Integer> singleSet = new HashSet<Integer>();
            singleSet.add(obj);
            combinations.add(singleSet);
        }
        return combinations;
    }

    public Map<Integer, List<HashSet<Integer>>> constructCombinations(){
        List<HashSet<Integer>> combinations;
        int numberOfFeatures = trainingSet.getTrainingData().get(0).attributes.size();
        List<Integer> featureIndexList = new ArrayList<Integer>();
        Map<Integer, List<HashSet<Integer>>> combinationPerSize = new HashMap<Integer, List<HashSet<Integer>>>();

        // Generate List
        for(int i=0; i < numberOfFeatures; i++){
            featureIndexList.add(i);
        }

        // Generate permutation
        combinations = getAllCombination(featureIndexList);
        for(HashSet<Integer> perSet: combinations){
            List<HashSet<Integer>> combPerSize = combinationPerSize.get(perSet.size());
            if(combPerSize == null){
                combPerSize = new ArrayList<HashSet<Integer>>();
                combPerSize.add(perSet);
            }
            else{
                combPerSize.add(perSet);
            }
            combinationPerSize.put(perSet.size(), combPerSize);
        }

        return combinationPerSize;
    }

    public List<Observation> copyTable(List<Observation> tableToCopy){
        List<Observation> newList = new ArrayList<Observation>();

        for(Observation obs: tableToCopy){
            newList.add(obs.clone());
        }
        return newList;
    }


    public void buildValuePerTableMatrix(){
        boolean cond;
        int nPerms = featureCombination.size();
        subTablePresent = new HashMap<HashSet<Integer>, Map<LinkedHashSet<Double>, List<Integer>>>();
        HashSet<Double> classes = new HashSet<Double>(trainingSet.getClasses());

        for(Double subClass: classes) {
            List<Observation> tmpSubList = subTableMap.get(subClass.intValue());
            for (int perm = 1; perm < nPerms; perm++) {
                List<HashSet<Integer>> localPermutation = featureCombination.get(perm);
                for(HashSet<Integer> singlePermutation : localPermutation) {

                    HashSet<LinkedHashSet<Double>> hsPerm;
                    List<List<Double>> perArray = new ArrayList<List<Double>>();

                    for (Integer idx : singlePermutation) {
                        List<Double> singleCol = getColumn(idx, tmpSubList);
                        perArray.add(singleCol);
                    }
                    List<LinkedHashSet<Double>> tmpHashSet = itemsPermutation(perArray, perArray.size() - 1);
                    hsPerm = new HashSet<LinkedHashSet<Double>>(tmpHashSet);
                    for(Iterator<LinkedHashSet<Double>> iter = hsPerm.iterator(); iter.hasNext();) {
                        LinkedHashSet<Double> obj = iter.next();
                        if(obj.size() != perm){
                            iter.remove();
                        }
                    }

                    for(LinkedHashSet<Double> sp: hsPerm){
                        for(Iterator<Observation> iter = tmpSubList.iterator(); iter.hasNext();) {
                            cond = true;
                            Observation obs = iter.next();
                            List<Integer> listPattern = new ArrayList<Integer>(singlePermutation);
                            List<Double> listElem = new ArrayList<Double>(sp);
                            for (int i = 0; i < listPattern.size(); i++) {
                                cond = cond && (obs.attributes.get(listPattern.get(i)).equals(listElem.get(i)));
                            }

                            if(cond){
                                Map<LinkedHashSet<Double>, List<Integer>> tmpMap =
                                        subTablePresent.get(singlePermutation);
                                if(tmpMap == null){
                                    tmpMap = new HashMap<LinkedHashSet<Double>, List<Integer>>();
                                    subTablePresent.put(singlePermutation, tmpMap);
                                }
                                List<Integer> tmpList = tmpMap.get(sp);
                                if(tmpList == null){
                                    tmpList = new ArrayList<Integer>();
                                    tmpMap.put(sp, tmpList);
                                }
                                tmpList.add(subClass.intValue());
                            }
                        }
                    }
                }
            }
        }
    }


    public boolean checkValueInOtherTables(HashSet<Double> permSet, HashSet<Integer> permutation, int actClass){
        LinkedHashSet<Double> linkedHash = new LinkedHashSet<Double>(permSet);
        List<Integer> classesPerPerm = subTablePresent.get(permutation).get(linkedHash);
        for(Integer subClass: classesPerPerm){
            if(subClass != actClass){
                return false;
            }
        }
        return true;
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


    public Pair findMostCommon(List<LinkedHashSet<Double>> hsPerm){
        int occur;
        int occurInd;
        LinkedHashSet<Double> mc;
        List<Integer> occurences = new ArrayList<Integer>();
        for(LinkedHashSet<Double> setHs: hsPerm){
            occurences.add(Collections.frequency(hsPerm, setHs));
        }
        occurInd = occurences.indexOf(Collections.max(occurences));
        occur = occurences.get(occurences.indexOf(Collections.max(occurences)));
        mc = hsPerm.get(occurInd);
        return new Pair(mc, occur);
    }

    public void train(){
        int perm;
        PatternPair maxPair;
        buildValuePerTableMatrix();
        rules = new ArrayList<PatternPair>();

        // Every Permutation
        for(Integer extClass: subTableMap.keySet()){
            perm = 1;
            List<Observation> localTable = copyTable(subTableMap.get(extClass));
            while(localTable.size() > 0){
                List<HashSet<Integer>> localPermutation = featureCombination.get(perm);
                List<PatternPair> maxCounts = new ArrayList<PatternPair>();
                for(HashSet<Integer> singlePermutation: localPermutation){
                    List<LinkedHashSet<Double>> hsPerm;

                    // Prepare are for counting permutation
                    List<List<Double>> perArray = new ArrayList<List<Double>>();
                    for(Integer idx: singlePermutation){
                        List<Double> singleCol = getColumn(idx, localTable);
                        perArray.add(singleCol);
                    }
                    hsPerm = itemsPermutation(perArray, perArray.size() - 1);
                    // Remove items from other lists
                    for(Iterator<LinkedHashSet<Double>> iter = hsPerm.iterator(); iter.hasNext();) {
                        LinkedHashSet<Double> hs = iter.next();
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

                    if(cond){
                        iter.remove();
                    }
                }
                rules.add(maxPair);
            }
        }
    }


}
