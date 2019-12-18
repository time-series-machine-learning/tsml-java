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
package tsml.examples;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import statistics.simulators.ArmaModel;
import statistics.simulators.DataSimulator;
import statistics.simulators.Model;
import statistics.simulators.SimulateSpectralData;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;

/**
 * Class demonstrating how to use the data simulators to generate weka instances
 * @author ajb
 */
public class DataSimulatorExamples extends DataSimulator{
    
    
    public static void main(String[] args) {
        Classifier c=new J48();
        try {
            c.buildClassifier(null);
        } catch (Exception ex) {
            Logger.getLogger(DataSimulatorExamples.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        
        try {
            c.buildClassifier(null);
        } catch (Exception ex) {  
            System.out.println("ERRRORRR!!!!");
            System.exit(0);
        }
        
        
        
/**DataSimulator: All the simulators inherit from  DataSimulator
 * a DataSimulator contains an ArrayList of Models, one for each class
 * To create a data simulator, you can either pass it a 2D array of parameters
 * (one array for each class) or pass it an ArrayList of models 
 * (again, one for each class).
*/
        double[][] paras={{0.1,0.5,-0.6},{0.2,0.4,-0.5}};
// Creates a two class simulator for AR(3) models         
        DataSimulator arma=new SimulateSpectralData(paras);
        
/* Model: All models inherit from the base Model class. Model has three abstract 
 * methods. generate: returns the next observation in the series, generate(t) 
 * generates the observation at time t (if possible) and generateSeries(int n), 
 * which calls generate n times and returns an array        */
        ArrayList<Model> m=new ArrayList<>();
        m.add(new ArmaModel(paras[0]));
        m.add(new ArmaModel(paras[1]));
   
/** Once you have created the simulator and/or the models, you can create sets 
 * of instances thus */
        int seriesLength=100;
        int[] casesPerClass={100,100};
        arma.setSeriesLength(seriesLength);
        arma.setCasesPerClass(casesPerClass);
        Instances data = arma.generateDataSet();
        
    }
}

