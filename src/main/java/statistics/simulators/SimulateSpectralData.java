/** Simulates AR parameters
 * 
 */


package statistics.simulators;
import timeseriesweka.filters.FFT;
import timeseriesweka.filters.ARMA;
import timeseriesweka.filters.PACF;
import timeseriesweka.filters.ACF;
import java.util.*;
import java.text.*;
//import utilities.OutFile;
import weka.core.*;
import fileIO.*;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import timeseriesweka.classifiers.FastDTW_1NN;
import weka.filters.*;





public class SimulateSpectralData extends DataSimulator{
    public static String path="C:\\Research\\Data\\RunLengthExperiments\\";
    public static double MIN_PARA_VALUE=-1;
    public static double MAX_PARA_VALUE=1;
    static int defaultMinSize=1;
    static int defaultMaxSize=4;
    double[][] parameters;
    int minParas=defaultMinSize;
    int maxParas=defaultMaxSize;
    static double DIFF=0.15;
    static double THRESH=0.95;
    public Random rand=new Random();
    
    public SimulateSpectralData(){
//Generate random model paras that are not trivial to classifier
        minParas=defaultMinSize;
        maxParas=defaultMaxSize;
        boolean validated=false;
        int maxRuns=0;
        int[] temp=casesPerClass;
        int temp2=nosPerClass;
        while(maxRuns<10 && !validated){
            parameters=generateModels();
            nosClasses=parameters.length;
            models=new ArrayList<Model>(nosClasses);
            for(int i=0;i<nosClasses;i++)
                models.add(new ArmaModel(parameters[i]));
            setCasesPerClass(new int[]{20,20});
            setLength(100);
            try{
            Instances[] data=this.generateTrainTest();
            Classifier c =new FastDTW_1NN();
            double acc=ClassifierTools.singleTrainTestSplitAccuracy(c, data[0], data[1]);
            if(acc<THRESH)
                validated=true;
            maxRuns++;
            }catch(Exception e){//Data generation failed
                maxRuns++;
                validated=false;
            }
        }
        casesPerClass=temp;
        nosPerClass=temp2;
        
    }
    public SimulateSpectralData(int seed){
        this();
        rand=new Random(seed);
    }
    
    public static void setMinMaxPara(double a, double b){MIN_PARA_VALUE=a; MAX_PARA_VALUE=b;}
//Default random model with DataSimulator defaults (2 classes seriesLength)    
    final public double[][] generateModels(){
        double[][] p=new double[nosClasses][];
        p[0]=generateStationaryParameters(minParas,maxParas);
//Try fixed perturbation
        for(int i=1;i<nosClasses;i++){
            p[i]=new double[p[0].length];
            for(int j=0;j<p[0].length;j++){
                if(rand.nextDouble()<0.5)
                    p[i][j]=p[i-1][j]-DIFF;
                else
                    p[i][j]=p[i-1][j]-DIFF;
                    if(p[i][j]<0)
                        p[i][j]=0;
                    else if(p[i][j]>1)
                        p[i][j]=1;
            }
        }
        return p;
    }
    
    public void randomiseModel(){
        parameters=new double[nosClasses][];
        for(int i=0;i<parameters.length;i++)
            parameters[i]=generateStationaryParameters(minParas,maxParas);
        nosClasses=parameters.length;
        models=new ArrayList<Model>(nosClasses);
        for(int i=0;i<nosClasses;i++)
            models.add(new ArmaModel(parameters[i]));
        
    }
    public void randomiseModel(int min,int max){
        minParas=min;
        maxParas=max;
        randomiseModel();
    }
    public void randomiseModel(int min,int max, int numClasses){
        nosClasses=numClasses;
        minParas=min;
        maxParas=max;
        randomiseModel();
    }
    public void randomiseModel(int min,int max, int numClasses, int length){
        nosClasses=numClasses;
        seriesLength=length;
        minParas=min;
        maxParas=max;
        randomiseModel();
    }
    public SimulateSpectralData(double[][] paras){
        super(paras);
        for(int i=0;i<nosClasses;i++)
            models.add(new ArmaModel(paras[i]));
    }
    public void initialise(){
        for(Model a:models)
                ((ArmaModel)a).randomise();
    }
    public double[] generate(int length, int modelNos){
            double[] d=new double[length];
            Model a=models.get(modelNos);
            for(int i=0;i<length;i++)
                    d[i]=a.generate();
            return d;
    }
 
/** Static test methods **/

    public static void testARMA(){
        double[][] paras;		
        int nosParas=(int)(Math.random()*11.0);
        DecimalFormat dc = new DecimalFormat("###.###");

        paras=new double[2][nosParas];
        for(int j=0;j<nosParas;j++)
                paras[0][j]=-.95+1.8*Math.random();
        System.out.print("\nInput coefficients");
        for(int i=0;i<paras[0].length;i++)
                System.out.print(dc.format(paras[0][i])+"\t");
        System.out.print("\n");
        paras[0]=findCoefficients(paras[0]);
        for(int j=0;j<nosParas;j++)
                paras[1][j]=paras[0][j]-0.1+0.2*Math.random();
        int n=200;
        double[] d1 = generate(paras[0],n);
        double[] d2 = generate(paras[0],n);
        for(int i=0;i<d1.length;i++)
                System.out.print(dc.format(d1[i])+"\t");
        double[] f1= ARMA.fitAR(d1);
        double[] f2= ARMA.fitAR(d2);
        System.out.println("\n\nModel length ="+nosParas);
        System.out.print("\nACTUAL MODEL 1=");
        for(int i=0;i<paras[0].length;i++)
                System.out.print(dc.format(paras[0][i])+"\t");
        System.out.print("\nFITTED MODEL 1=");
        for(int i=0;i<f1.length;i++)
                System.out.print(dc.format(f1[i])+"\t");
        System.out.println("\n\nModel length ="+nosParas);
        System.out.print("\nACTUAL MODEL 2=");
        for(int i=0;i<paras[0].length;i++)
                System.out.print(dc.format(paras[1][i])+"\t");
        System.out.print("\nFITTED MODEL 2=");
        for(int i=0;i<f2.length;i++)
                System.out.print(dc.format(f2[i])+"\t");

    }
    
    public static Instances generateSpectralEmbeddedData(int s,int[] casesPerClass){
        DataSimulator ds=new SimulateSpectralData();
        ds.setCasesPerClass(casesPerClass);
        int fullLength=s;
        int arLength=fullLength/2;

        ds.setLength(arLength);
        Instances data=ds.generateDataSet();
        NormalizeCase nc= new NormalizeCase();
        try{
            data=nc.process(data);
        }catch(Exception e){
            System.out.println("Normalisation failed : "+e);
        }
        ArrayList<Model> noise=new ArrayList<>();
        WhiteNoiseModel wm=new WhiteNoiseModel();
        noise.add(wm);
        wm=new WhiteNoiseModel();
        noise.add(wm);
        DataSimulator ds2=new DataSimulator(noise); // By default it goes to white noise 
        ds2.setCasesPerClass(casesPerClass);
        ds2.setLength(fullLength);
        Instances noiseData=ds2.generateDataSet();
        
//Choose random start
        int startPos=(int)(Math.random()*(fullLength-arLength));
        for(int j=startPos;j<startPos+arLength;j++){
            for(int k=0;k<data.numInstances();k++)
                noiseData.instance(k).setValue(j, data.instance(k).value(j-startPos));
        }
        return noiseData;
    }
    
    public static void testFFT(String fileName){
        //Debug code to test. 
        //Generate a model and series
//			double[][] paras={{0.5},{0.7}};
        int n=32;
        int[] cases={1,1};
        FFT ar = new FFT();
        double[][] paras={{1.3532,0.4188,-1.2153,0.3091,0.1877,-0.0876,0.0075,0.0004},
        {1.0524,0.9042,-1.2193,0.0312,0.263,-0.0567,-0.0019}	};	
        Instances train=SimulateSpectralData.generateARDataSet(paras,n,cases);
        //Fit and compare paramaters without AIC
        OutFile of = new OutFile(fileName);
        try{
                DecimalFormat dc=new DecimalFormat("###.####");
                Instances arTrain=ar.process(train);
                Instance in1=train.instance(0);
                System.out.print("\nFitted Data Full >\t");
                for(int i=0;i<in1.numAttributes()-1;i++){
                        of.writeString(in1.value(i)+",");
                        System.out.print(in1.value(i)+",");
                }
                of.writeString("\n");
                Instance in2=arTrain.instance(0);
                System.out.print("\nFitted FFT Full >\t");
                for(int i=0;i<in2.numAttributes();i++){
                        System.out.print(dc.format(in2.value(i))+",");
                        of.writeString(dc.format(in2.value(i))+",");
                }
                ar.truncate(arTrain, n/4);
                System.out.print("\nFitted FFT Truncated >\t");
                of.writeString("\n");
                for(int i=0;i<in2.numAttributes();i++){
                        System.out.print(dc.format(in2.value(i))+",");
                        of.writeString(dc.format(in2.value(i))+",");
                }

        }catch(Exception e){
                System.out.println("Exception ="+e);
                e.printStackTrace();
                System.exit(0);
        }
    }		
    public static void testAIC(String fileName){
        //Debug code to test. 
        //Generate a model and series
//			double[][] paras={{0.5},{0.7}};
        int n=100;
        int[] cases={1,1};
        ARMA ar = new ARMA();
        ar.setUseAIC(false);
        int maxLag=n/4;
        ar.setMaxLag(maxLag);
        double[][] paras={{1.3532,0.4188,-1.2153,0.3091,0.1877,-0.0876,0.0075,0.0004},
        {1.0524,0.9042,-1.2193,0.0312,0.263,-0.0567,-0.0019}	};	
        Instances train=SimulateSpectralData.generateARDataSet(paras,n,cases);
        //Fit and compare paramaters without AIC
        try{
                DecimalFormat dc=new DecimalFormat("###.####");
                Instances arTrain=ar.process(train);
                Instance in1=train.instance(0);
                Instance in2=arTrain.instance(0);
                System.out.print("Actual Model >\t\t");
                for(int i=0;i<paras[0].length;i++)
                        System.out.print(dc.format(paras[0][i])+",");
                System.out.print("\nFitted Model No AIC >\t");
                for(int i=0;i<in2.numAttributes();i++)
                        System.out.print(dc.format(in2.value(i))+",");
                ar.setUseAIC(true);
                arTrain=ar.process(train);
                in2=arTrain.instance(0);
                System.out.print("\nFitted Model AIC >\t");
                for(int i=0;i<in2.numAttributes();i++)
                        System.out.print(dc.format(in2.value(i))+",");
                //Debug the stages
                OutFile of = new OutFile(fileName);
                for(int i=0;i<in1.numAttributes()-1;i++)
                        of.writeString(in1.value(i)+",");
                double[] d=in1.toDoubleArray();
                double[] d2=new double[d.length-1];
                for(int i=0;i<d2.length;i++)
                        d2[i]=d[i];
                System.out.println("Auto Corellations >");
                double[] autos=ACF.fitAutoCorrelations(d2,maxLag);
                of.writeString("\n");
                of.writeString("\n");
                for(int i=0;i<autos.length;i++){
                        of.writeString(autos[i]+",");
                        System.out.println(autos[i]+",");
                }
                of.writeString("\n");
                of.writeString("\n");
                double[][] partials=PACF.formPartials(autos);	
                for(int i=0;i<partials.length;i++){
                        for(int j=0;j<partials[i].length;j++)
                                of.writeString(partials[i][j]+",");
                        of.writeString("\n");
                }
                int best=ARMA.findBestAIC(autos,partials,maxLag,d2);
                System.out.println(" Best Length = "+best);

        }catch(Exception e){
                System.out.println("Exception ="+e);
                System.exit(0);
        }
    }

/** Static methods to bypass object creation
 * 
 
 */
/**
 * This generates a set of 2-Class AR 1 data sets with fixed size and varying differences 
 * between the two class models. Needs calibrating for complexity. 
 * 
 * Noise is always N(0,1).
 * 
 * Range of values initially, model 1: 0.9
 * model 2: 0.9 to -0.9 in 0.05 increments
 */
 
    public static void generateAR1(int n, double[][] p, String fileName){
        DecimalFormat df = new DecimalFormat("####.####");
        int nosCases=200;
        double diff=0.05;
        String newline=System.getProperty("line.separator");
        SimulateSpectralData ar= new SimulateSpectralData(p);
        String arffHeader="@relation AR1Models\n";
        for(int i=0;i<n;i++)
                arffHeader+="@attribute T"+i+" real"+newline;
        arffHeader+="@attribute targetClass {0,1}"+newline;
        arffHeader+="@data"+newline+newline;
        OutFile of =new OutFile(path+"AR1\\train"+fileName);
        of.writeString(arffHeader);
        for(int i=0;i<nosCases;i++){
                ar.initialise();
                double[] d=ar.generate(n,0);
                for(int j=0;j<d.length;j++)
                        of.writeString(df.format(d[j])+",");
                of.writeString("0"+newline);
                d=ar.generate(n,1);
                for(int j=0;j<d.length;j++)
                        of.writeString(df.format(d[j])+",");
                of.writeString("1"+newline);
        }
        of =new OutFile(path+"AR1\\test"+fileName);
        of.writeString(arffHeader);
        for(int i=0;i<nosCases;i++){
                ar.initialise();
                double[] d=ar.generate(n,0);
                for(int j=0;j<d.length;j++)
                        of.writeString(df.format(d[j])+",");
                of.writeString("0\n");
                d=ar.generate(n,1);
                for(int j=0;j<d.length;j++)
                        of.writeString(df.format(d[j])+",");
                of.writeString("1\n");
        }			
    }
    public static double[] findCoefficients(double[] c){
        int n=c.length;
        double[] a=new double[1];
        double[] aNew=null;
        a[0]=1;
//			System.out.println(" n = "+n);
        for(int j=1;j<=n;j++){
//				System.out.println("Finding order "+j);
            aNew=new double[j+1];
            aNew[0]=1;
            for(int i=1;i<j;i++)
                    aNew[i]=a[i]-a[i-1]*c[j-1];
            aNew[j]=a[j-1]*-c[j-1];
            a=aNew;
        }
//Remove the constant term
        double[] f=new double[n];
        for(int i=0;i<n;i++)
                f[i]=-a[i+1];
        return f;
    }
		
    public static double[] generate(double[] paras,int n){
            double[] d = new double[n];
            ArmaModel ar=new ArmaModel(paras);
            ar.randomise();
            for(int i=0;i<n;i++)
                    d[i]=ar.generate();
            return d;

    }
                
/** generateStationaryParameters
 * Generates a random AR model of degree between minParas and maxParas inclusive
 * 
 * 1. Generate a number between minParas and maxParas
 * 2. Generate random numbers between -0.9 and 0.9
 * 3. find the stationary AR parameters associated with these parameters
 */
    public static double[] generateStationaryParameters(int minP, int maxP){
            double[] paras;
            int nosParas=(int)(minP+(int)(1+maxP*Model.rand.nextDouble()));
                paras=new double[nosParas];
            for(int j=0;j<nosParas;j++)
                    paras[j]=MIN_PARA_VALUE+(MAX_PARA_VALUE-MIN_PARA_VALUE)*Model.rand.nextDouble();
            paras=SimulateSpectralData.findCoefficients(paras);
            return paras;
    }
                
    public static void armaTest(){
        double[][] paras;		
        int nosParas=(int)(Model.rand.nextDouble()*11.0);
        DecimalFormat dc = new DecimalFormat("###.###");

        paras=new double[2][nosParas];
        for(int j=0;j<nosParas;j++)
                paras[0][j]=-.95+1.8*Model.rand.nextDouble();
        System.out.print("\nInput coefficients");
        for(int i=0;i<paras[0].length;i++)
                System.out.print(dc.format(paras[0][i])+"\t");
        System.out.print("\n");
        paras[0]=findCoefficients(paras[0]);
        for(int j=0;j<nosParas;j++)
                paras[1][j]=paras[0][j]-0.1+0.2*Model.rand.nextDouble();
        int n=200;
        double[] d1 = generate(paras[0],n);
        double[] d2 = generate(paras[0],n);
        for(int i=0;i<d1.length;i++)
                System.out.print(dc.format(d1[i])+"\t");
        double[] f1= ARMA.fitAR(d1);
        double[] f2= ARMA.fitAR(d2);
        System.out.println("\n\nModel length ="+nosParas);
        System.out.print("\nACTUAL MODEL 1=");
        for(int i=0;i<paras[0].length;i++)
                System.out.print(dc.format(paras[0][i])+"\t");
        System.out.print("\nFITTED MODEL 1=");
        for(int i=0;i<f1.length;i++)
                System.out.print(dc.format(f1[i])+"\t");
        System.out.println("\n\nModel length ="+nosParas);
        System.out.print("\nACTUAL MODEL 2=");
        for(int i=0;i<paras[0].length;i++)
                System.out.print(dc.format(paras[1][i])+"\t");
        System.out.print("\nFITTED MODEL 2=");
        for(int i=0;i<f2.length;i++)
                System.out.print(dc.format(f2[i])+"\t");

    }  
    public static Instances generateOffByOneARDataSet(int p, int seriesLength, int[] nosCases, boolean normalize){
        double[][] paras=new double[nosCases.length][];
        paras[0]=generateStationaryParameters(p,p);
        for(int i=1;i<paras.length;i++){
            paras[i]=generateStationaryParameters(p+i,p+i);
        }
        Instances d=generateARDataSet(paras,seriesLength,nosCases);
        if(normalize){
        try{
            NormalizeCase norm=new NormalizeCase();
            norm.setNormType(NormalizeCase.NormType.STD_NORMAL);
                d=norm.process(d);
            }catch(Exception e){
                System.out.println("Exception e"+e);
                e.printStackTrace();
                System.exit(0);
            }
        }
        return d;
    }    

        public static Instances generateARDataSet(int seriesLength, int[] nosCases, boolean normalize){
            return generateARDataSet(defaultMinSize,defaultMaxSize,seriesLength,nosCases,normalize);
            
        }

        
        
    public static Instances generateARDataSet(int minParas, int maxParas, int seriesLength, int[] nosCases, boolean normalize){
        double[][] paras=new double[nosCases.length][];
        for(int i=0;i<paras.length;i++)
            paras[i]=generateStationaryParameters(minParas,maxParas);
        Instances d=generateARDataSet(paras,seriesLength,nosCases);
        if(normalize){
        try{
            NormalizeCase norm=new NormalizeCase();
            norm.setNormType(NormalizeCase.NormType.STD_NORMAL);
                d=norm.process(d);
            }catch(Exception e){
                System.out.println("Exception e"+e);
                e.printStackTrace();
                System.exit(0);
            }
        }
        return d;
    }
    public static Instances generateARDataSet(int minParas, int maxParas, int seriesLength, int[] nosCases){
        return generateARDataSet(minParas,maxParas,seriesLength,nosCases,false);
    }                

    public static Instances generateARDataSet(double[][] p, int seriesLength, int[] nosCases){
        SimulateSpectralData ar=new SimulateSpectralData(p);
        Instances data;
        FastVector atts=new FastVector();
        int totalCases=nosCases[0];
        for(int i=1;i<nosCases.length;i++)
                totalCases+=nosCases[i];
        for(int i=1;i<=seriesLength;i++){
                atts.addElement(new Attribute("t"+i));
        }
        FastVector fv=new FastVector();
        for(int i=0;i<ar.nosClasses;i++)
                fv.addElement(""+i);
        atts.addElement(new Attribute("Target",fv));
        data = new Instances("AR",atts,totalCases);

        double[] d;
        for(int i=0;i<ar.nosClasses;i++){
            for(int j=0;j<nosCases[i];j++){
//Generate the series					
                ar.initialise();
                d=ar.generate(seriesLength,i);
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
                
    public static Instances generateFFTDataSet(int minParas, int maxParas, int seriesLength, int[] nosCases, boolean normalize){
        double[][] paras=new double[nosCases.length][];
//Generate random parameters for the first FFT        
        Random rand= new Random();
        SinusoidalModel[] sm=new SinusoidalModel[nosCases.length];
        int modelSize=minParas+rand.nextInt(maxParas-minParas);
        paras[0]=new double[3*modelSize];
        for(int j=0;j<paras.length;j++)
             paras[0][j]=rand.nextDouble();
        for(int i=1;i<sm.length;i++){
            paras[i]=new double[3*modelSize];
            for(int j=0;j<paras.length;j++){
                paras[i][j]=paras[0][j];
//Perturb it 10%
                paras[i][j]+=-0.1+0.2*rand.nextDouble();
                if(paras[i][j]<0 || paras[i][j]>1)
                    paras[i][j]=paras[0][j];
            }
        }
        for(int i=0;i<sm.length;i++){
            sm[i]=new SinusoidalModel(paras[i]);
            sm[i].setFixedOffset(false);
        }
            
        
//        for(int i=0;i<paras.length;i++)
//            paras[i]=generateStationaryParameters(minParas,maxParas);
        DataSimulator ds = new DataSimulator(sm);
        ds.setSeriesLength(seriesLength);
        ds.setCasesPerClass(nosCases);
        Instances d=ds.generateDataSet();
        if(normalize){
        try{
            NormalizeCase norm=new NormalizeCase();
            norm.setNormType(NormalizeCase.NormType.STD_NORMAL);
            d=norm.process(d);
            }catch(Exception e){
                System.out.println("Exception e"+e);
                e.printStackTrace();
                System.exit(0);
            }
        }
        return d;
    }
                
    
    
    
    public static void main(String[] args){
//                    Instances all=SimulateSpectralData.generateARDataSet(minParas,maxParas,seriesLength,nosCasesPerClass,true);
        System.out.println("Running SimulateAR test harness");
        ArmaModel.setGlobalVariance(1);
        int[] cases={2,2};
        Instances d=generateARDataSet(1,1,200,cases,false);
        ArmaModel.setGlobalVariance(10);
        Instances d2=generateARDataSet(1,1,200,cases,false);
        OutFile of=new OutFile("C:\\\\Research\\Results\\AR1_1.csv");
        OutFile of2=new OutFile("C:\\\\Research\\Results\\AR1_10.csv");
        of.writeString(d.toString());
        of2.writeString(d2.toString());

        try{
            NormalizeCase norm=new NormalizeCase();
            norm.setNormType(NormalizeCase.NormType.STD_NORMAL);
                d=norm.process(d);
            }catch(Exception e){
                System.out.println("Exception e"+e);
                e.printStackTrace();
                System.exit(0);
       }
    }
   @Override
    public String getParameters() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

}
