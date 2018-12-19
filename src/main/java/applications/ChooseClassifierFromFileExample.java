/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package applications;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import vector_classifiers.ChooseClassifierFromFile;
import weka.core.Instances;

/**
 *
 * @author cjr13geu
 */
public class ChooseClassifierFromFileExample {
    
    static File[] relationNames; 
    
    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {
        
        ChooseClassifierFromFile cCFF = new ChooseClassifierFromFile();
        
        FileReader arffReader = new FileReader("../test.ARFF");
        
        Instances instances = new Instances(arffReader);
         
        String classifiers[] = {"TunedSVMRBF", "TunedSVMPolynomial"};
        
        cCFF.setResultsPath("\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Results\\UCIContinuous\\");
        
        cCFF.setClassifiers(classifiers);

        cCFF.setName("TunedSVM");
        
        //cCFF.setRelationName("chess-krvk");
        
        File relationFolder = new File("\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Results\\UCIContinuous\\TunedSVMRBF\\Predictions");
        
        relationNames = relationFolder.listFiles();
        
        for (int i = 0; i < relationNames.length; i++) {
            cCFF.setRelationName(relationNames[i].getName());
            for (int j = 0; j < 30; j++) {
                cCFF.setFold(j);
                cCFF.buildClassifier(instances);
            }
        } 
    }
    
}
