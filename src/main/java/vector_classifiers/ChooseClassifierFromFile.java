/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package vector_classifiers;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author cjr13geu
 */    
public class ChooseClassifierFromFile implements Classifier{
    
    private Random randomNumber;
    private final int bufferSize = 100000;
    private int foldNumber = 0;
    private int indexOfLargest = 0;
    ArrayList<String> line;
    
    /**
     * if size results path == 1, all classifier's results read from that one path
     * else, resultsPaths.length must equal classifiers.length, with each index aligning 
     * to the path to read the classifier's results from.
     * 
     * e.g to read 2 classifiers from one directory, and another 2 from 2 different directories: 
     * 
     *     Index |  Paths  | Classifier
     *     --------------------------
     *       0   |  pathA  |   c1
     *       1   |  pathA  |   c2
     *       2   |  pathB  |   c3
     *       3   |  pathC  |   c4 
     * 
     */
    private String[] resultsPaths = { "Results/" }; 
    
    /**
     * if resultsWritePath is not set, will default to resultsPaths[0]
     * i.e, if only reading from one directory, will write back the chosen results 
     * under the same directory. if reading from multiple directories but a particular 
     * write path not set, will simply pick the first one given. 
     */
    private String resultsWritePath = null;
    private String classifiers[] = {"TunedSVMRBF", "TunedSVMPolynomial"};
    
    private String name = "EnsembleResults";  
    private String relationName = "abalone";
    private double accuracies[];
    private File dir;
    private BufferedReader[] trainFiles;
    private BufferedReader testFile;
    private BufferedWriter outTrain;
    private BufferedWriter outTest;
    
    
    
    public void setFold(int foldNumber){
        this.foldNumber = foldNumber;
    }
    
    public void setClassifiers(String[] classifiers){
        this.classifiers = classifiers;
    }
    
    public void setResultsPath(String[] resultsPaths){
        this.resultsPaths = resultsPaths;
    }
    
    public void setResultsPath(String resultsPath){
        this.resultsPaths = new String[] { resultsPath };
    }
    
    public void setResultsWritePath(String writePath) {
        this.resultsWritePath = writePath;
    }
    
    public void setName(String name){
        this.name = name;
    }
    
    public void setRelationName(String name){
        this.relationName = name;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (resultsPaths.length > 1)
            if (resultsPaths.length != classifiers.length)
                throw new Exception("ChooseClassifierFromFile.buildClassifier: more than one results path given, but number given does not align with the number of classifiers.");
        
        if (resultsWritePath == null)
            resultsWritePath = resultsPaths[0];
        
        dir = new File(resultsWritePath + "/" + this.name + "/Predictions/" + relationName + "/trainFold" + foldNumber + ".csv");
        
        if(!dir.exists()){
            try{ 
                trainFiles = new BufferedReader[classifiers.length];
                accuracies = new double[classifiers.length];

                for (int i = 0; i < classifiers.length; i++) {
                    int pathIndex = resultsPaths.length == 1 ? 0 : i;
                    
                    trainFiles[i] = new BufferedReader(new FileReader(resultsPaths[pathIndex] + "/"+ classifiers[i] + "/Predictions/" + relationName + "/trainFold" + foldNumber + ".csv"), bufferSize);
                    trainFiles[i].mark(bufferSize);
                    trainFiles[i].readLine();
                    trainFiles[i].readLine();
                    accuracies[i] = Double.valueOf(trainFiles[i].readLine());
                }

                for (int i = 0; i < accuracies.length; i++ ) { 
                    if ( accuracies[i] > accuracies[indexOfLargest] ) { 
                        indexOfLargest = i; 
                    }
                }

                ArrayList<Integer> duplicates = new ArrayList<>();
                for (int i = 0; i < accuracies.length; i++) {
                    if(accuracies[indexOfLargest] == accuracies[i] && indexOfLargest != i){
                        duplicates.add(i);
                    }
                }

                randomNumber = new Random(foldNumber);
                if(!duplicates.isEmpty()){
                    indexOfLargest = randomNumber.nextInt(duplicates.size());
                }

                //Write Train file.
                dir = new File(resultsWritePath + "/" + this.name + "/Predictions/" + relationName);
                dir.mkdirs();
                outTrain = new BufferedWriter(new FileWriter(dir + "/trainFold" + foldNumber + ".csv"));
                trainFiles[indexOfLargest].reset();
                line = new ArrayList<>(Arrays.asList(trainFiles[indexOfLargest].readLine().split(",")));
                line.set(1, name);
                outTrain.write(line.toString().replace("[", "").replace("]", ""));
                outTrain.newLine();

                line = new ArrayList<>(Arrays.asList(trainFiles[indexOfLargest].readLine().split(",")));
                line.add("originalClassifier");
                line.add(classifiers[indexOfLargest]);
                outTrain.write(line.toString().replace("[", "").replace("]", ""));
                outTrain.newLine();

                while((line = new ArrayList<>(Arrays.asList(new String[] { trainFiles[indexOfLargest].readLine() }))).get(0)  != null){
                    outTrain.write(line.get(0));
                    outTrain.newLine();
                }

                //Write Test file.
                outTest = new BufferedWriter(new FileWriter(dir + "/testFold" + foldNumber + ".csv"));
                
                int pathIndex = resultsPaths.length == 1 ? 0 : indexOfLargest;
                testFile = new BufferedReader(new FileReader(resultsPaths[pathIndex] + "/"+ classifiers[indexOfLargest] + "/Predictions/" + relationName + "/testFold" + foldNumber + ".csv"), bufferSize);
                
                line = new ArrayList<>(Arrays.asList(testFile.readLine().split(",")));
                line.set(1, name);
                outTest.write(line.toString().replace("[", "").replace("]", ""));
                outTest.newLine();

                line = new ArrayList<>(Arrays.asList(testFile.readLine().split(",")));
                line.add("originalClassifier");
                line.add(classifiers[indexOfLargest]);
                outTest.write(line.toString().replace("[", "").replace("]", ""));
                outTest.newLine();

                while((line = new ArrayList<>(Arrays.asList(new String[] { testFile.readLine() }))).get(0)  != null){
                    outTest.write(line.get(0));
                    outTest.newLine();
                }


                for (int i = 0; i < classifiers.length; i++) {
                    trainFiles[i].close();
                    testFile.close();
                }
                outTrain.flush();
                outTrain.close();
                outTest.flush();
                outTest.close();

            }catch(FileNotFoundException | NumberFormatException e){
                System.out.println("Fold " + foldNumber + " not present: "+ e);
            }
        }else{
            System.out.println(dir.getAbsolutePath() + ": Already exists.");
        }
        
        
    } 

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        classifyInstance(instance);
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}

