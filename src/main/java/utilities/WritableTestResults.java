/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package utilities;

import java.io.File;
import java.io.FileWriter;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public interface WritableTestResults extends Classifier{

    default void writeTestResultsToFile(Instances testData, String datasetName, String classifierName, String paramLine, String outputFilePathAndName) throws Exception{
        
        new File(outputFilePathAndName).getParentFile().mkdirs();
        FileWriter out = new FileWriter(outputFilePathAndName);
        out.close();
        if(new File(outputFilePathAndName).exists()==false){
            throw new Exception("Error: could not create file "+outputFilePathAndName);
        }
        
        
        int correct = 0;
        double actual, pred;
        double[] dists;
        double bsfClass;
        double bsfWeight;
        StringBuilder lineBuilder;
        StringBuilder outBuilder = new StringBuilder();
        
        for(int i = 0; i < testData.numInstances(); i++){
            actual = testData.instance(i).classValue();
            dists = this.distributionForInstance(testData.instance(i));
            
            lineBuilder = new StringBuilder();
            bsfClass = -1;
            bsfWeight = -1;
            
            for(int c = 0; c < dists.length; c++){
                if(dists[c] > bsfWeight){
                    bsfWeight = dists[c];
                    bsfClass = c;
                }
                lineBuilder.append(",").append(dists[c]);
            }
            if(bsfClass==actual){
                correct++;
            }
            outBuilder.append(actual).append(",").append(bsfClass).append(",").append(lineBuilder.toString()).append("\n");
        }
        out = new FileWriter(outputFilePathAndName);
        out.append(datasetName+","+classifierName+",test\n");
        out.append(paramLine+"\n");
        out.append(((double)correct/testData.numInstances())+"\n");
        out.append(outBuilder.toString());
        out.close();
    }
}
