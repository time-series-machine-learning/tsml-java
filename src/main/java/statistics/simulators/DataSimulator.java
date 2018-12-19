/*
 Base class for data simulator. Three use cases. 

1. Set up models externally then call generateData
ArrayList<Model> m = ....
DataSimulator ds = new DataSimulator(m);
Instances data=ds.generateData();

2. Use a subclass of DataSimulator
DataSimulator ds = new SimulateShapeletDataset();
Instances data=ds.generateData();



 */
package statistics.simulators;

import java.util.ArrayList;
import java.util.Arrays;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.NormalizeCase;

/**
 *
 * @author ajb
 */
public class DataSimulator {
    int nosClasses=2;
    ArrayList<Model> models;
// * @param seriesLength Length of each series, assumed same for all instances
    int seriesLength=100;
    int nosPerClass=50;
// * @param casesPerClass. nosCases.length specifies the number of cases (which should already be stored), casesPerClass[i] gives the number of cases in class i 
    int[] casesPerClass;
    public void setSeriesLength(int s){
        seriesLength=s;
    }
    public void setCasesPerClass(int[] c){
        casesPerClass=c;
        nosPerClass=c[0];   //Assumed equal number per class
    }
//Override this to define a default random model    
    protected DataSimulator(){
        models=new ArrayList<Model>();
                
    }
    protected DataSimulator(double[][] paras){
        nosClasses=paras.length;
        models=new ArrayList<Model>(nosClasses);
    } 
/**
 * So you can either load the models like this, or you can subclass 
 * DataSimulator and redefine the constructor
 * to create the base Models. This is a deep copy, I THINK!?
 * @param m 
 */
    public DataSimulator(ArrayList<Model> m){
        models=new ArrayList<>();
        nosClasses=m.size();          
        models.addAll(m);
    }
    @SuppressWarnings("ManualArrayToCollectionCopy")
        public DataSimulator(Model[] m){
            nosClasses=m.length;
            models=new ArrayList<Model>(nosClasses);
            for(int i=0;i<m.length;i++)
                models.add(m[i]);
        }
        public void setModel(ArrayList<Model> m){
            nosClasses=m.size();
            models.addAll(m);
        }
        public ArrayList<Model> getModels(){ return models;}
    @SuppressWarnings("ManualArrayToCollectionCopy")
        public void setModel(Model[] m){
            nosClasses=m.length;
            for(int i=0;i<m.length;i++)
                models.add(m[i]);
        }
/**
 * @PRE: All parameters of the model have been set through other means
 * @POST: no change to the model, no instances are stored
 * 
 * @return Set of n=sum(casesPerClass[i]) instances, each seriesLength+1 attributes, the last of which is the class label,
 */
    public Instances generateDataSet() {
        
        Instances data;
        if(casesPerClass==null){    
            casesPerClass=new int[nosClasses];
            for(int i=0;i<casesPerClass.length;i++)
                casesPerClass[i]=nosPerClass;
        }
        FastVector atts=new FastVector();
        nosClasses=casesPerClass.length;
        int totalCases=casesPerClass[0];
        for(int i=1;i<casesPerClass.length;i++)
                totalCases+=casesPerClass[i];
        for(int i=1;i<=seriesLength;i++){
                atts.addElement(new Attribute(models.get(0).getAttributeName()+i));
        }
        FastVector fv=new FastVector();
        for(int i=0;i<nosClasses;i++)
                fv.addElement(""+i);
        atts.addElement(new Attribute("Target",fv));
        data = new Instances(models.get(0).getModelType(),atts,totalCases);

        double[] d;
        for(int i=0;i<nosClasses;i++){
            for(int j=0;j<casesPerClass[i];j++){
//Generate the series					
                initialise();
                d=generate(seriesLength,i);
//Add to an instance
                Instance in= new DenseInstance(data.numAttributes());
                for(int k=0;k<d.length;k++)
                        in.setValue(k,d[k]);
//Add to all instances					
                data.add(in);
                in=data.lastInstance();
                in.setValue(d.length,""+i);
            }

        }
        data.setClassIndex(seriesLength);
        
        return data;
    }
    
/**
 * @PRE: All parameters of the model have been set through other means
 * @POST: no change to the model, no instances are stored
**/
    public String generateHeader(){
        String header="%"+"  "+models.get(0).getModelType()+"\n"; 
        for(int i=0;i<models.size();i++){
            header+="%Class "+i;
            header+="\n"+models.get(i).getHeader()+"\n";
        }
        return header;
    }
    
    public Instances[] generateTrainTest() throws Exception{
        Instances[] data=new Instances[2];
        data[0]=generateDataSet();
//        initialise();//Rest models? depends if the model is deterministic! might cause some problems either way
        data[1]=generateDataSet();
 //Normalise
        NormalizeCase nc= new NormalizeCase();
        data[0]=nc.process(data[0]);
        data[1]=nc.process(data[1]);
            return data;
        
        
    }
    
    
    public double[] generate(int length, int modelNos){
        double[] d=new double[length];
        Model a=models.get(modelNos);
        d=a.generateSeries(length);
        return d;
    }
/** 
 * This method
 */    
    public void initialise(){
        for(Model a:models)
            a.reset();
    }
    public void setNosPerClass(int x){
        nosPerClass=x;
    }
    public void setLength(int l){
        seriesLength=l;
    }
    
/**
 * @return String with all parameter names and values
 */    
   public String getParameters(){
       String str=nosClasses+"\n";
       for(Model m:models)
           str+=m.toString()+"\n";
       return str;
   } 
}
