package com.naivebayes;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by filip on 18.10.16.
 */

class DataFile {

    private BufferedReader bufferedReader;

    // DataFile Constructor
    public DataFile(String dataResource){
        try {
            ClassLoader loader = DataFile.class.getClassLoader();
            bufferedReader = new BufferedReader(
                    new FileReader(loader.getResource(dataResource).getFile())
            );

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    // Get single observation
    public List<Double> getObservation() throws IOException {
        String observationLine;
        String[] splitedString;
        List<Double> resultRecord = new ArrayList<Double>();

        if ((observationLine = bufferedReader.readLine()) != null) {
            splitedString = observationLine.split(",");
        }
        else{
            return null;
        }

        // Parse String array to Double list
        for (String aSplitedString : splitedString) {
            resultRecord.add(Double.parseDouble(aSplitedString));
        }

        return resultRecord;
    }

    // Get Data Set
    public List<List<Double>> getDataset() throws IOException {
        List<List<Double>> observationSet = new ArrayList<List<Double>>();
        List<Double> signleObservation;

        while((signleObservation = getObservation()) != null){
            observationSet.add(signleObservation);
        }

        return observationSet;

    }

}
