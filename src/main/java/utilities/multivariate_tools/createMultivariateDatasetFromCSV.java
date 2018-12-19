/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities.multivariate_tools;

import static utilities.multivariate_tools.ConvertDatasets.buildArff;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import weka.core.Instances;

/**
 *
 * @author Aaron
 */
public class createMultivariateDatasetFromCSV {
    
    
    public static void main(String[] args) throws FileNotFoundException {

        File f = new File("D:\\Work\\Dropbox\\Multivariate TSC\\Baydogan Matlab\\");
        for(File f1 : f.listFiles()){
            String name = f1.getName();
            createDataset(f1.getAbsolutePath()+"\\", name, name+"_train.csv", "TRAIN");
            createDataset(f1.getAbsolutePath()+"\\",name, name+"_test.csv", "TEST");
        }
        
    }
    
    static void createDataset(String dir, String name, String dataset_csv, String affix) throws FileNotFoundException{
                
        //extract dataset into a matrix and class value format.
        List<Data> dataset = loadDataset(new File(dir + dataset_csv));
    
        //need to find the shortest series in the dataset and then prune all the other series to that length.
        int shortest_length = dataset.stream().mapToInt(e -> e.mat.stream()
                                                              .mapToInt(e1 -> e1.length)
                                                              .min().getAsInt())
                                               .min().getAsInt();
  
        //trim the arrays,
        for(Data data : dataset)
            for(int i=0; i<data.mat.size(); i++)
                data.mat.set(i, Arrays.copyOfRange(data.mat.get(i), 0, shortest_length));

        //then convert the List<Data> format to ARFF. might be easy to do univariate files and then combine?
        //build arff creates an Instances file from a list of double[] arrays and an array of class values. 
        //buildArff
        Instances[] univariate_datasets = new Instances[dataset.get(0).mat.size()];
        for(int i=0; i<univariate_datasets.length; i++){
            List<double[]> series = new ArrayList();
            double[] labels = new double[dataset.size()];
            for(int j=0; j<dataset.size(); j++){
                series.add(dataset.get(j).mat.get(i));
                labels[j] = dataset.get(j).val;
            }
            
            univariate_datasets[i] = buildArff(series,labels);
        }
        
        for(int i=0; i<univariate_datasets.length; i++)
            utilities.ClassifierTools.saveDataset(univariate_datasets[i], dir + name + "_" + i + "_"+ affix);
            
        Instances output = utilities.multivariate_tools.MultivariateInstanceTools.mergeToMultivariateInstances(univariate_datasets);
        utilities.ClassifierTools.saveDataset(output, dir + name + "_"+ affix);
    }
    
    static List<Data> loadDataset(File fname) throws FileNotFoundException{
        //load the file
        
        Scanner sc = new Scanner(fname);
        //load each line until a line is only size one. which must be the class value.
        
        List<Data> dataset = new ArrayList<Data>();
        List<double[]> mat = new ArrayList();
                
        while(sc.hasNextLine()){
            String line = sc.nextLine();
            //val and new line char
            if(line.length()>2){
                double[] doubleValues = Arrays.stream(line.split(","))
                        .mapToDouble(Double::parseDouble)
                        .toArray();
                mat.add(doubleValues); 
            }else{
                dataset.add(new Data(mat, Double.parseDouble(line)));
                mat = new ArrayList();
            }
        }
        return dataset;
    }
    
    static class Data{
        List<double[]> mat;
        double val;
        
        Data(List<double[]> matrix, double classValue){
            mat = matrix;
            val = classValue;
        }
    }
}

