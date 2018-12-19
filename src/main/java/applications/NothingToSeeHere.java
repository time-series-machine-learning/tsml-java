/*
 just somewhere to test general code, of no interes
 */
package applications;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class NothingToSeeHere extends Thread{
    
    
 public enum Coin {FIVEPENCE(5),TENPENCE(10),
                TWENTYPENCE(20),FIFTYPENCE(50),ONEPOUND(100);
    private int v;
    private Coin(int val){v=val;}
    public int getValue(){return v;}
 
 }


public static int sumValue(ArrayList<Coin> ar){
    int sum=0;
    for(Coin a:ar)
        sum+= a.getValue();
    return sum;


}   
   

//Test the MLP results    
    public static void testMLPSettings() throws Exception{
        MultilayerPerceptron mlp=new MultilayerPerceptron();
        String problem="hayes-roth";
        Instances all=ClassifierTools.loadData("//cmptscsvr.cmp.uea.ac.uk/ueatsc/Data/UCIContinuous/"+problem+"/"+problem);
        Instances[] split=InstanceTools.resampleInstances(all, 0, 0.5);
        Instances train=split[0];
        Instances test=split[1];
//Default acc
        double a =ClassifierTools.singleTrainTestSplitAccuracy(mlp, train, test);
        System.out.println("Default Acc"+a);
        System.out.println(" PARAS: Learning rate "+mlp.getLearningRate()+" momentum = "+mlp.getMomentum()+" Decay = "+mlp.getDecay()+" Network ="+mlp.getHiddenLayers());
        mlp=new MultilayerPerceptron();
        mlp.setLearningRate(0.0625);
        mlp.setMomentum(0);
        mlp.setDecay(false);
        mlp.setHiddenLayers("t");
        a =ClassifierTools.singleTrainTestSplitAccuracy(mlp, train, test);
        System.out.println("Best Acc"+a);
         System.out.println(" PARAS: Learning rate "+mlp.getLearningRate()+" momentum = "+mlp.getMomentum()+" Decay = "+mlp.getDecay()+" Network ="+mlp.getHiddenLayers());
       
//Configure MLP
//Nodes	t	LearningRate	0.0625	Momentum	0	Decay	FALSE
//	CVAcc	0.759493671	Test Acc 0.444444444

        
    }
    
    public void run(){
        List<Double> d= new ArrayList<Double>();
                InFile f;
                
    }
    public static void main(String[] args) throws Exception {
                testMLPSettings();
        OutFile out = new OutFile("PermissionsTest.csv");
        out.writeLine("TestyTestytest");
        out.closeFile();
        File f=new File("PermissionsTest.csv");
        f.setWritable(true,false);
       
    }
}
