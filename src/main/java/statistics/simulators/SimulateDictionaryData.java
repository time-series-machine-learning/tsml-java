/*
Simulate repeated patterns in the data
 */
package statistics.simulators;

import fileIO.OutFile;
import utilities.InstanceTools;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class SimulateDictionaryData {
    static int[] shapeletsPerClass={5,20};//Also defines the num classes by length
    static int shapeLength=11;
//Store a global copy purely to be able to recover the latest metadata
//Probably need to generalise this1?  
    static DataSimulator sim;
    public static void setShapeletsPerClass(int[] c){
        shapeletsPerClass=new int[c.length];
        for(int i=0;i<c.length;i++)
           shapeletsPerClass[i]=c[i]; 
    }
    public static void setShapeletLength(int s){
        shapeLength=s;
    }
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
    public static Instances generateDictionaryData(int seriesLength, int []casesPerClass)
    {        
        if( casesPerClass.length != 2)
        {
            System.err.println("Incorrect parameters, dataset will not be co"
                    + "rrect.");
            int[] tmp = {0,0};
            casesPerClass = tmp;
            
        }
        DictionaryModel[] shapeMod = new DictionaryModel[casesPerClass.length];
        populateRepeatedShapeletArray(shapeMod, seriesLength);
//        for(DictionaryModel s:shapeMod)
//            System.out.println("Shapel Model "+s);
        sim = new DataSimulator(shapeMod);
        sim.setSeriesLength(seriesLength);
        sim.setCasesPerClass(casesPerClass);
        Instances d=sim.generateDataSet();
        return d;          
    }
   /**
     * This is a support method for generateShapeletData
     * 
     * @param array An array of two ShapeletModel2 models, representing the 
     * simulated shapes inserted into the respective classes.
     * @param seriesLength The length of the series.
     */
    private static void populateRepeatedShapeletArray(DictionaryModel [] s, int seriesLength)
    {
        if(s.length!=shapeletsPerClass.length){//ERROR
            throw new RuntimeException("Error, mismatch in number of classes: "+s.length+" VS "+shapeletsPerClass.length);
        }
        for(int i=0;i<s.length;i++){
            double[] p1={seriesLength,shapeletsPerClass[(0+i)],shapeletsPerClass[(1+i)%2],shapeLength};
            s[i]=new DictionaryModel(p1);        
        }
//Fix all the shape types to be the same as first
        DictionaryModel.ShapeType st = s[0].getShape1();
        for(int i=1;i<s.length;i++)
            s[i].setShape1Type(st);
        st = s[0].getShape2();
        for(int i=1;i<s.length;i++)
            s[i].setShape2Type(st);
    }

    public static void generateExampleData(){
        int seriesLength=1000;
        Model.setDefaultSigma(0.1);
        Instances noNoise=generateDictionaryData(seriesLength,new int[]{20,20});

        Model.setDefaultSigma(1);
        Instances noise=generateDictionaryData(seriesLength,new int[]{20,20});
        OutFile of = new OutFile("C:\\temp\\DictionaryNoNoise.csv");
        of.writeString(noNoise.toString());
        OutFile of2 = new OutFile("C:\\temp\\DictionaryNoise.csv");
        of2.writeString(noise.toString());
        
    }    
    public static void checkGlobalSeedForIntervals(){
        Model.setDefaultSigma(0);
        Model.setGlobalRandomSeed(0);
        Instances d=generateDictionaryData(100,new int[]{2,2});
        OutFile of = new OutFile("C:\\Temp\\randZeroNoiseSeed0.csv");
       of.writeLine(d.toString());
        Model.setDefaultSigma(0.1);
        Model.setGlobalRandomSeed(1);
         d=generateDictionaryData(100,new int[]{2,2});
        of = new OutFile("C:\\Temp\\randUnitNoiseSeed1.csv");
       of.writeLine(d.toString());
        Model.setDefaultSigma(0);
        Model.setGlobalRandomSeed(0);
         d=generateDictionaryData(100,new int[]{2,2});
        of = new OutFile("C:\\Temp\\randZeroNoiseSeed0REP.csv");
       of.writeLine(d.toString());
        Model.setDefaultSigma(0.1);
        Model.setGlobalRandomSeed(1);
         d=generateDictionaryData(100,new int[]{2,2});
        of = new OutFile("C:\\Temp\\randUnitNoiseSeed1REP.csv");
       of.writeLine(d.toString());
    }
    public static void main(String[] args) {
        checkGlobalSeedForIntervals();
        System.exit(0);
//        generateExampleData();
        Model.setDefaultSigma(0);
        Model.setGlobalRandomSeed(0);
//seriesLength=1000;
//                casesPerClass=new int[]{20,20};        
        Instances d=generateDictionaryData(59,new int[]{2,2});
        Instances[] split=InstanceTools.resampleInstances(d, 0,0.5);
        System.out.println(" DATA "+d);
        OutFile of = new OutFile("C:\\Temp\\dictionarySimulationTest.csv");
//        of.writeLine(""+sim.generateHeader());
        of.writeString(split[0].toString());
        of = new OutFile("C:\\Temp\\dictionarySimulationTrain.csv");
        of.writeString(split[1].toString());
    }
        
}
