/*

 */
package statistics.simulators;

import development.DataSets;
import fileIO.OutFile;
import utilities.InstanceTools;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class SimulateIntervalData {
    /**
     * This method creates and returns a set of Instances representing a
     * simulated two-class time-series problem.
     * 

     * @param seriesLength The length of the series. All time series in the
     * dataset are the same length.
     * @param casesPerClass An array of two integers indicating the number of 
     * instances of class 0 and class 1.
     * @return Instances representing the time-series dataset. The Instances
     * returned will be empty if the casesPerClass parameter does not contain
     * exactly two values.
     */
    static DataSimulator sim;
    static int intervalLength;    
    static int nosIntervals=3; 
//The pro    
    static int noiseToSignal=4;
    public static void setNosIntervals(int n){ nosIntervals=n;}
    public static void setNoiseToSignal(int n){ noiseToSignal=n;}
    static double base=-1;
    static double amp=2;
    public static void setAmp(double a){
        amp=a;
        base=-amp/2;
    }
    public static Instances generateIntervalData(int seriesLength, int []casesPerClass)
    {
        int s=Model.getRandomSeed();
        //        OutFile model=new OutFile(DataSets.clusterPath+"temp/model"+s+".csv");
        intervalLength=seriesLength/(nosIntervals*noiseToSignal);
        DictionaryModel.Shape.DEFAULTBASE=base;
        DictionaryModel.Shape.DEFAULTAMP=amp;
/*        model.writeString("IN SimulateIntervalData \n SeriesLength,"+seriesLength+",classes,"+casesPerClass.length+",");
        for(int x:casesPerClass)
            model.writeString(","+x);
            model.writeString("\nbase,"+base+","+amp+",SeriesLength,"+seriesLength+",intLength,"+intervalLength+",NosIntervals,"+nosIntervals);
 */       
        IntervalModel[] intervalMod = new IntervalModel[casesPerClass.length];
        populateIntervalModels(intervalMod,seriesLength); 
/*        model.writeString("\nIN IntervalModel\n");        
        model.writeLine("Model 1:\n"+intervalMod[0].toString());
        model.writeLine("Model 2:\n"+intervalMod[1].toString());
        model.closeFile();
*/        sim = new DataSimulator(intervalMod);
        sim.setSeriesLength(seriesLength);
        sim.setCasesPerClass(casesPerClass);
        Instances d=sim.generateDataSet();
        return d;
    }
    private static void populateIntervalModels(IntervalModel[] m, int seriesLength){
        if(m.length!=2)
            System.out.println("ONLY IMPLEMENTED FOR TWO CLASSES");
//Create two models with same interval but different shape. 
        IntervalModel m1=new IntervalModel();
        m1.setSeriesLength(seriesLength);
        m1.setNoiseToSignal(noiseToSignal);
        m1.setNosIntervals(nosIntervals);
        m1.createIntervals();
        IntervalModel m2=new IntervalModel();
        m2.setSeriesLength(seriesLength);
        m1.setNoiseToSignal(noiseToSignal);
        m2.setNosIntervals(nosIntervals);
        m2.setIntervals(m1.getIntervals(), m1.getIntervalLength());
//Set shapes for intervals. Start by having same shape on each interval, but  
//different shape per model        
//Thi may give an advantage to the spectral classifiers, could have a different shape 
        m1.randomiseShape(null);
        m2.randomiseShape(m1);
        m[0]=m1;
        m[1]=m2;
        
    }
    public static void main(String[] args) {
        Model.setDefaultSigma(0);
        Model.setGlobalRandomSeed(0);
        int seriesLength=500;
        int[] casesPerClass=new int[]{2,2};        
        SimulateIntervalData.setNosIntervals(4);
        SimulateIntervalData.setNoiseToSignal(4);
        Instances d=generateIntervalData(seriesLength,casesPerClass);
        Instances[] split=InstanceTools.resampleInstances(d, 0,0.5);
        System.out.println(" DATA "+d);
        OutFile of = new OutFile("C:\\Temp\\intervalSimulationTest.csv");
//        of.writeLine(""+sim.generateHeader());
        of.writeString(split[0].toString());
        of = new OutFile("C:\\Temp\\intervalSimulationTrain.csv");
        of.writeString(split[1].toString());
    }
    
}
