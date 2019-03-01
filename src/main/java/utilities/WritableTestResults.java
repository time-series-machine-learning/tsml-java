/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
