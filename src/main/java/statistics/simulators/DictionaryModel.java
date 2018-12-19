/*
AJB Oct 2016

Model to simulate data where dictionary approach should be optimal.

A single shapelet is common to all series.  The discriminatory feature is the 
number of times it appears in a series. 


*/
package statistics.simulators;
import fileIO.OutFile;
import java.util.*;
import java.io.*;
import statistics.distributions.NormalDistribution;

public class DictionaryModel extends Model {
    
    public enum ShapeType {TRIANGLE,HEADSHOULDERS,SINE, STEP, SPIKE};
    private static int DEFAULTNUMSHAPELETS=5;
    private static int DEFAULTSERIESLENGTH=1000;
    public static int DEFAULTSHAPELETLENGTH=29;

    
    protected Shape shape1;
    protected Shape shape2;
    protected int numShape1=DEFAULTNUMSHAPELETS;
    protected int[] shape1Locations=new int[numShape1];
    protected int numShape2=DEFAULTNUMSHAPELETS;
    protected int[] shape2Locations=new int[numShape2];
    
    protected int totalNumShapes=numShape1+numShape2;
    protected int seriesLength=DEFAULTSERIESLENGTH; 
    protected int shapeletLength=DEFAULTSHAPELETLENGTH;

    //Default Constructor, max start should be at least 29 less than the length
    // of the series if using the default shapelet length of 29
    public DictionaryModel()
    {
        this(new double[]{DEFAULTSERIESLENGTH,DEFAULTNUMSHAPELETS,DEFAULTNUMSHAPELETS,DEFAULTSHAPELETLENGTH});
    }
    public DictionaryModel(double[] param)
    {
        super();
        setDefaults();
//PARAMETER LIST: seriesLength,  numShape1, numShape2, shapeletLength
        if(param!=null){
            switch(param.length){
                default: 
                case 4:             shapeletLength=(int)param[3];
                case 3:             numShape2=(int)param[2];
                case 2:             numShape1=(int)param[1];
                case 1:             seriesLength=(int)param[0];
            }
        }
        totalNumShapes=numShape1+numShape2;
        shape1Locations=new int[numShape1];
        shape2Locations=new int[numShape2];
        shape1=new Shape(shapeletLength);
//        shape1.type=ShapeType.TRIANGLE;
        shape1.length=shapeletLength;
        shape1.randomiseShape();
        shape2=new Shape(shapeletLength);
        shape2.randomiseShape();
        while(shape2.type==shape1.type)
            shape2.randomiseShape();
//Enforce non-overlapping, only occurs if there is not enough room for 
//all the shapes. Locations split randomly between the two classes
//This is reset with a call to generateSeries        
        while(!setNonOverlappingLocations()){
            totalNumShapes--;
            if(numShape1>numShape2)
                numShape1--;
            else
                numShape2--;
        }
        
    }

    public final void setDefaults(){
       seriesLength=DEFAULTSERIESLENGTH; 
       numShape1=DEFAULTNUMSHAPELETS;
       shapeletLength=DEFAULTSHAPELETLENGTH;
    }
    public ShapeType getShape1(){
        return shape1.type;
    }
    public ShapeType getShape2(){
        return shape2.type;
    }
    public void setShape1Type(ShapeType st){
        shape1.setType(st);
        shape1.setLength(shapeletLength);
    }
    public void setShape2Type(ShapeType st){
        shape2.setType(st);
        shape2.setLength(shapeletLength);
    }
   
    public void setNumShape1(int n){
        numShape1=n;
    }
    public void setNumShape2(int n){
        numShape2=n;
    }
    final public boolean setNonOverlappingLocations(){
        if(seriesLength-shapeletLength*totalNumShapes<totalNumShapes)  //Cannot fit them in, not enough spaces
            return false;
//Me giving up and just randomly placing the shapes until they are all non overlapping
        ArrayList<Integer>  locations=new ArrayList<>();
        for(int i=0;i<totalNumShapes;i++){
            boolean ok=false;
            int l=shapeletLength/2;
            while(!ok){
                ok=true;
//Search mid points to level the distribution up somewhat                
                l=rand.nextInt(seriesLength-shapeletLength)+shapeletLength/2;
  //          System.out.println("trying   "+l);
                
                for(int in:locations){
                    if((l>=in-shapeletLength && l<in+shapeletLength) //l inside ins
                      ||(l<in-shapeletLength && l+shapeletLength>in)      ){ //ins inside l
                        ok=false;
//                        System.out.println(l+"  overlaps with "+in);
                        break;
                    }
                }
            }
//           System.out.println("Adding "+l);
            locations.add(l);
        }
//Revert to start points            
            for(int i=0;i<locations.size();i++){
//Just in case ..
                int val=locations.get(i);
                locations.set(i, val-shapeletLength/2);
            }
//            System.out.println("Location ="+(l-shapeletLength/2));
        
        shape1Locations=new int[numShape1];
       for(int i=0;i<numShape1;i++)
           shape1Locations[i]=locations.get(i);
        shape2Locations=new int[numShape2];
       for(int i=0;i<numShape2;i++)
           shape2Locations[i]=locations.get(i+numShape1);
        Arrays.sort(shape1Locations);
        Arrays.sort(shape2Locations);
       
        return true;
    }
    final public boolean setNonOverlappingLocationsOld(){
//        if(seriesLength-shapeletLength*totalNumShapes<totalNumShapes)  //Cannot fit them in, not enough spaces
//          return false;
        int s=0;
//        while(s)
   //http://stackoverflow.com/questions/33831442/random-placement-of-non-overlapping-intervals
      
//Find non overlapping locations. 
        
        
//1.        Specify how many spaces there are
        int spaces=seriesLength-shapeletLength*totalNumShapes+1;
//We now need to randomly distribute these spaces between each shapelet.        
        int nosIntervals=totalNumShapes+1;
//Split spaces into nosIntervals
        ArrayList<Integer>  intervals=new ArrayList<>();
/*        for(int i=0;i<nosIntervals;i++){
            int r=rand.nextInt(spaces-(nosIntervals-i));
            intervals.add(r);
            spaces-=r;
        }
*/
        int[] temp=new int[nosIntervals];
        for(int i=0;i<spaces;i++)
            temp[new Random().nextInt(nosIntervals)]++;
        for(int i=0;i<nosIntervals;i++)
            intervals.add(temp[i]);
//Randomize intervals
//       Collections.shuffle(intervals, rand);
//Add back into place        
        ArrayList<Integer>  shapeLocations=new ArrayList<Integer> (totalNumShapes);
        shapeLocations.add(intervals.get(0));
        int current=shapeLocations.get(0)+shapeletLength;
        for(int i=1;i<totalNumShapes;i++){
           shapeLocations.add(current+intervals.get(i));
           current=shapeLocations.get(i)+shapeletLength;
       }
//Randomize  locations
       Collections.shuffle(shapeLocations, rand);
       
      
//Split randomised locations for two types of shape
       for(int i=0;i<numShape1;i++)
           shape1Locations[i]=shapeLocations.get(i);
       for(int i=0;i<numShape2;i++)
           shape2Locations[i]=shapeLocations.get(i+numShape1);
       
       Arrays.sort(shape1Locations);
       Arrays.sort(shape2Locations);

/*
//Sort them       
       Arrays.sort(shapeLocations);
//Shift forward if necessary
       for(int i=1;i<shapeLocations.length;i++){
           if(shapeLocations[i]<shapeLocations[i-1]+shapeletLength)
               shapeLocations[i]=shapeLocations[i-1]+shapeletLength;
       }
//Randomise again
       intervals=new ArrayList<Integer>();
       for(int i:shapeLocations)
           intervals.add(i);
       Collections.shuffle(intervals, rand);
       
//Split randomised locations for two types of shape
       for(int i=0;i<numShape1;i++)
           shape1Locations[i]=intervals.get(i);
       for(int i=0;i<numShape2;i++)
           shape2Locations[i]=intervals.get(i+numShape1);
       
       Arrays.sort(shape1Locations);
       Arrays.sort(shape2Locations);
//Find the positions by reflating
/*       
       
        int count1=0;
        int count2=0;
//Try without alwayspushing forward        
        
        for(int i=0;i<totalNumShapes;i++){
            if(count2==shape2Locations.length) //Finished shape2, must do shape1
               shape1Locations[count1++]+=i*shapeletLength;
            else if(count1==shape1Locations.length) //Finished shape1, must do shape1
               shape2Locations[count2++]+=i*shapeletLength;
            else if(shape1Locations[count1]<shape2Locations[count2])//Shape 1 before Shape 2, inflate shape 1
               shape1Locations[count1++]+=i*shapeletLength;
            else //Inflate shape2
               shape2Locations[count2++]+=i*shapeletLength;
        }
  */      
        return true;
    }
    
    /*Generate a single data
//Assumes a model independent of previous observations. As
//such will not be relevant for ARMA or HMM models, which just return -1.
* Should probably remove. 
*/
    @Override
	public double generate(double x){
//Noise
            int t=(int)x;
            double value=error.simulate();
//Shape: Check if in a shape1
 /*           int insertionPoint=Arrays.binarySearch(shapeLocations,t);
                        if(insertionPoint<0)//Not a start pos: in
                insertionPoint=-(1+insertionPoint);
//Too much grief, just doing a linear scan!            
            */
//See if it is in shape1            
            int insertionPoint=0;
            while(insertionPoint<shape1Locations.length && shape1Locations[insertionPoint]+shapeletLength<t)
                insertionPoint++;    
            if(insertionPoint>=shape1Locations.length){ //Bigger than all the start points, set to last
                insertionPoint=shape1Locations.length-1;
            }
            if(shape1Locations[insertionPoint]<=t && shape1Locations[insertionPoint]+shapeletLength>t){//in shape1
                value+=shape1.generateWithinShapelet(t-shape1Locations[insertionPoint]);
//                System.out.println(" IN SHAPE 1 occurence "+insertionPoint+" Time "+t);
            }else{  //Check if in shape 2
                insertionPoint=0;
                while(insertionPoint<shape2Locations.length && shape2Locations[insertionPoint]+shapeletLength<t)
                    insertionPoint++;    
                if(insertionPoint>=shape2Locations.length){ //Bigger than all the start points, set to last
                    insertionPoint=shape2Locations.length-1;
                }
                if(shape2Locations[insertionPoint]<=t && shape2Locations[insertionPoint]+shapeletLength>t){//in shape2
                    value+=shape2.generateWithinShapelet(t-shape2Locations[insertionPoint]);
//                System.out.println(" IN SHAPE 2 occurence "+insertionPoint+" Time "+t);
                }
            }
            return value;
        }

//This will generateWithinShapelet the next sequence after currently stored t value
    @Override
	public double generate()
        {
//            System.out.println("t ="+t);
            double value=generate(t);
            t++;
            return value;
        }
     @Override   
	public	double[] generateSeries(int n)
	{
           t=0;
//Resets the starting locations each time this is called          
           setNonOverlappingLocations();
           double[] d = new double[n];
           for(int i=0;i<n;i++)
              d[i]=generate();
           return d;
        }
    
   
    
    /**
 * Subclasses must implement this, how they take them out of the array is their business.
 * @param p 
 */ 
    @Override
    public void setParameters(double[] param){
        if(param!=null){
            switch(param.length){
                default: 
                case 4:             shapeletLength=(int)param[3];
                case 3:             numShape2=(int)param[2];
                case 2:             numShape1=(int)param[1];
                case 1:             seriesLength=(int)param[0];
            }
        }
         
        
    }
    @Override
    public String getModelType(){ return "DictionarySimulator";}
    @Override
        public String getAttributeName(){return "Dict";} 
    @Override
        public String getHeader(){
            String header=super.getHeader();
            header+="%  \t Shapelet Length ="+shapeletLength;
            header+="\n%  \t Series Length ="+seriesLength;
            header+="\n%  \t Number of Shapelets ="+numShape1;
            header+="\n% \t Shape = "+shape1.type;
            return header;
        }
    
    
        
    // Inner class determining the shape1 inserted into the shapelet model
    public static class Shape{
        // Type: head and shoulders, spike, step, triangle, or sine wave.
        public ShapeType type;
        //Length of shape1
        public int length;
        //Position of shape1 on axis determined by base (lowest point) and amp(litude).
        private double base;
        private double amp;
        //The position in the series at which the shape1 begins.
//        private int location;
        public void setBase(double b){ base=b;}
        public void setAmp(double a){amp=a;}
        public double getBase(){ return base;}
        public double getAmp(){return amp;}
        public static double DEFAULTBASE=-2;
        public static double DEFAULTAMP=4;
        
        //Default constructor, call randomise shape1 to get a random instance
        // The default length is 29, the shape1 extends from -2 to +2, is of 
        // type head and shoulders, and is located at index 0.
        public Shape(){
            this(ShapeType.HEADSHOULDERS,DEFAULTSHAPELETLENGTH,DEFAULTBASE,DEFAULTAMP); 
            if(type==ShapeType.HEADSHOULDERS)
                base=base/2;
            
        }  
        //Set length only, default for the others
        public Shape(int length){
            this(ShapeType.HEADSHOULDERS,length,DEFAULTBASE,DEFAULTAMP);      
            if(type==ShapeType.HEADSHOULDERS)
                base=base/2;
         }       
        public Shape(int l, double b, double a){
            randomiseShape();
            length=l;
            base=b;
            amp=a;
        }
        // This constructor produces a completely specified shape1
        public Shape(ShapeType t,int l, double b, double a){
            type=t;
            length=l;
            base=b;
            amp=a;
            if(type==ShapeType.HEADSHOULDERS)
                base=base/2;
        }
        public void setLength(int newLength){
            this.length=newLength;
        }
         
//Generates the t^th shapelet position
        public double generateWithinShapelet(int offset){
            double value=0;
            int lower=0,mid=0,upper=0;
//            if(offset==0)
//                System.out.println("LENGTH ="+length+" TYPE ="+type);
            switch(type){
             case TRIANGLE:
                 mid=length/2;
                 if(offset<=mid) {
                    if(offset==0)
                       value=base;
                    else
                       value=((offset/(double)(mid))*(amp))+base;
                 }
                 else
                 {
                     if(offset>=length)
                         value=base;
                     else if(length%2==1)
                         value=((length-offset-1)/(double)(mid)*(amp))+base;
                     else
                         value=((length-offset)/(double)(mid)*(amp))+base;
                 }
                   break;
                case HEADSHOULDERS:
//Need to properly set the boundaries for shapelets of length not divisible by 3. 
                  lower=length/3;
                  upper=2*lower;
//                    Do something about uneven split. 
                if(length%3==2)    //Add two the middle hump, or one each to the sides? 
                    upper+=2;
                if(offset<lower)//First small hump.
                    value = ((amp/2)*Math.sin(((2*Math.PI)/((length/3-1)*2))*offset))+base;
                else if(offset>=upper)//last small hump
                    value = ((amp/2)*Math.sin(((2*Math.PI)/((length/3-1)*2))*(offset-upper)))+base;
                else // middle big hump. Need to rescale for middle interval 
                        value = ((amp)*Math.sin(((2*Math.PI)/(((upper-lower)-1)*2))*(offset-length/3)))+base;
                if(value< base)//Case for length%3==1
                    value=base;
                   break;                
                case SINE:
                     value=amp*Math.sin(((2*Math.PI)/(length-1))*offset)/2;
                break;
                case STEP:
                    if(offset<length/2)
                        value=base;
                    else
                        value=base+amp;
                    break;
                case SPIKE:
               lower=length/4;
                upper=3*lower;
                if(offset<=lower)    //From 0 to base
                 {
                    if(offset==0)
                       value=0;
                    else
                       value=(-amp/2)*(offset/(double)(lower));
                 }
                else if(offset>lower && offset<upper)
                {
                        value=-amp/2+(amp)*((offset-lower)/(double)(upper-lower-1));
//                if(offset>length/2&&offset<=length/4*3)
//                    value=(offset-length/2)/(double)(length/4)*(amp/2);
                }                               
                else{
                      value=amp/2-amp/2*((offset-upper+1)/(double)(length-upper));
                 }
                break;
            }
            return value;
        }        
        
        public void setType(ShapeType newType){
            this.type=newType;
            if(newType==ShapeType.HEADSHOULDERS)
                base=DEFAULTBASE/2;
            else
                base=DEFAULTBASE;
        }
        
        
        @Override
        public String toString(){
            String shp = ""+this.type+",length,"+this.length+",this.base,"+base+",amp,"+this.amp;
            return shp;
        }
        
        //gives a shape1 a random type and start position
        public void randomiseShape(){
            ShapeType [] types = ShapeType.values();            
            int ranType = Model.rand.nextInt(types.length);
            setType(types[ranType]);
                
        }
        @Override
        public boolean equals(Object o){
           if(! (o instanceof Shape)) return false;
           if(type == ((Shape)o).type) return true;
           return false;
        }
         
    }
//1. CHECK PROPERLY REPRODUCABLE RANDOM
    public static void testRandSeed(){
       Model.setDefaultSigma(.1);
       Model.setGlobalRandomSeed(0);
       DEFAULTSHAPELETLENGTH=29;
       DEFAULTSERIESLENGTH=100;
       double[][] d=new double[ShapeType.values().length][DEFAULTSERIESLENGTH];
       int j=0;
       for(ShapeType s:ShapeType.values()){
//           DEFAULTSHAPELETLENGTH+=2;
//seriesLength,  numShape1, numShape2, shapeletLength           
           DictionaryModel shape = new DictionaryModel(new double[]{DEFAULTSERIESLENGTH,1,1,DEFAULTSHAPELETLENGTH});
           shape.setShape1Type(s);
           shape.setShape2Type(s);
           System.out.println(" SHAPE ="+s);
           for(int i=0;i<DEFAULTSERIESLENGTH;i++)
                d[j][i]=shape.generate(i);
           j++;
        }
    
       OutFile out=new OutFile("C:\\temp\\dictNoNoiseRep1.csv");
      for(int i=0;i<DEFAULTSERIESLENGTH;i++){
          for(j=0;j<d.length;j++)
            out.writeString(d[j][i]+",");
          out.writeString("\n");
      }
        
    }
    public static void testRandomisedPlacement(){
//2. CHECK LOCATIONS ARE RANDOM
       Model.setDefaultSigma(.1);
       Model.setGlobalRandomSeed(0);
       DEFAULTSHAPELETLENGTH=29;
       DEFAULTSERIESLENGTH=100;
           OutFile of = new OutFile("C:\\temp\\locationsDict.csv");
           of.writeString(",Shape1,,Shape2");
       for(int i=0;i<10000;i++){
           DictionaryModel shape = new DictionaryModel(new double[]{DEFAULTSERIESLENGTH,1,1,DEFAULTSHAPELETLENGTH});
           shape.setNonOverlappingLocations();
           for(int j=0;j<shape.numShape1;j++){
               of.writeString(","+shape.shape1Locations[j]);
           }
           of.writeString(",");
           for(int j=0;j<shape.numShape2;j++){
               of.writeString(","+shape.shape2Locations[j]);
           }
           of.writeString("\n");
           
       }
        
    }
    
    public static void main (String[] args) throws IOException
    {
//3. CHECK RESIZED SHAPELETS WORKS        
       Model.setDefaultSigma(0);
       Model.setGlobalRandomSeed(0);
       int sLength=30;
       int length=100;
       Shape s=new Shape(sLength);
       DictionaryModel shape;
              int c=0;
       double[][] data=new double[10][];
       for( sLength=10;sLength<=10;sLength+=4){
//           length=2*sLength+2;
       OutFile of = new OutFile("C:\\temp\\headShoulders"+sLength+".csv");
           System.out.println("SLength = "+sLength+" Length ="+length);
           shape = new DictionaryModel(new double[]{length,2,1,sLength});
//           shape.setShape1Type(ShapeType.HEADSHOULDERS);
//           shape.setShape2Type(ShapeType.HEADSHOULDERS);
           Model.setGlobalRandomSeed(4);
           
           for(int i=0;i<data.length;i++){
                Model.setGlobalRandomSeed(i);
               data[i]=shape.generateSeries(length);
           }
           for(int j=0;j<length;j++){
               for(int i=0;i<data.length;i++)
                 of.writeString(data[i][j]+",");
               of.writeString("\n");
           }
           }        
 
//4. Check multiple shapelet placement       
       
//5. To do in Generator class: CHECK RESTART WORKS PROPERLY, generating new random positions        
//       shape1.reset();
       
//       for(int i=0;i<200;i++)
//           System.out.println(shape1.generate(i));

         
    }
        
    
}
