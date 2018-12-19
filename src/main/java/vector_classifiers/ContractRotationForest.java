/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    RotationForest.java
 *    Copyright (C) 2008 Juan Jose Rodriguez
 *    Copyright (C) 2008 University of Waikato, Hamilton, New Zealand
 *
 */


package vector_classifiers;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayList;
import weka.classifiers.meta.*;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Randomizable;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.WeightedInstancesHandler;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.util.Enumeration;
import java.util.LinkedList;
import java.util.Random;
import java.util.Vector;
import timeseriesweka.classifiers.CheckpointClassifier;
import timeseriesweka.classifiers.ContractClassifier;
import utilities.ClassifierResults;
import utilities.SaveParameterInfo;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;


public class ContractRotationForest extends AbstractClassifier

  implements SaveParameterInfo, ContractClassifier, CheckpointClassifier, Serializable{
  
    Classifier baseClassifier;
    ArrayList<Classifier> classifiers;
    /** for serialization */
    static final long serialVersionUID = -3255631880798499936L;
    /** The minimum size of a group */
    protected int minGroup = 3;
    /** The maximum size of a group */
    protected int maxGroup = 3;
    /** The percentage of instances to be removed */
    protected int removedPercentage = 50;
    /** The attributes of each group */
    ArrayList< int[][]> groups;
    /** The type of projection filter */
    protected Filter projectionFilter;
    /** The projection filters */
    protected ArrayList<Filter []> projectionFilters;
    /** Headers of the transformed dataset */
    protected ArrayList<Instances> headers;
    /** Headers of the reduced datasets */
    protected ArrayList<Instances []> reducedHeaders;
    /** Filter that remove useless attributes */
    protected RemoveUseless removeUseless = null;
    /** Filter that normalized the attributes */
    protected Normalize normalize = null;

//Added features
    double contractHours=1;    //Defaults to an approximate build time of 1 hour
    protected ClassifierResults res;
    double estSingleTree;
    int numTrees=0;
    int minNumTrees=50;
    int maxNumTrees=200;
    int maxNumAttributes;
    String checkpointPath;
    boolean debug=true;
    TimingModel tm;
    double timeUsed;
    Random random;
    int seed =0;
    double alpha=0.2;//Learning rate for timing update

  /**
   * Constructor.
   */
  public ContractRotationForest() {
    
    baseClassifier = new weka.classifiers.trees.J48();
    projectionFilter = defaultFilter();
    tm=new TimingModel();
    res=new ClassifierResults();
    checkpointPath=null;
    timeUsed=0;
    
  }

  /**
   * Default projection method.
   */
  protected Filter defaultFilter() {
    PrincipalComponents filter = new PrincipalComponents();
    //filter.setNormalize(false);
    filter.setVarianceCovered(1.0);
    return filter;
  }
  public void setSeed(int s){
      seed=s;
  }

  /**
   * Sets the minimum size of a group.
   *
   * @param minGroup the minimum value.
   * of attributes.
   */
  public void setMinGroup( int minGroup ) throws IllegalArgumentException {

    if( minGroup <= 0 )
      throw new IllegalArgumentException( "MinGroup has to be positive." );
    this.minGroup = minGroup;
  }

  /**
   * Gets the minimum size of a group.
   *
   * @return 		the minimum value.
   */
  public int getMinGroup() {
    return minGroup;
  }

 
  /**
   * Sets the maximum size of a group.
   *
   * @param maxGroup the maximum value.
   * of attributes.
   */
  public void setMaxGroup( int maxGroup ) throws IllegalArgumentException {
 
    if( maxGroup <= 0 )
      throw new IllegalArgumentException( "MaxGroup has to be positive." );
    this.maxGroup = maxGroup;
  }

  /**
   * Gets the maximum size of a group.
   *
   * @return 		the maximum value.
   */
  public int getMaxGroup() {
    return maxGroup;
  }

  

  /**
   * Sets the percentage of instance to be removed
   *
   * @param removedPercentage the percentage.
   */
  public void setRemovedPercentage( int removedPercentage ) throws IllegalArgumentException {

    if( removedPercentage < 0 )
      throw new IllegalArgumentException( "RemovedPercentage has to be >=0." );
    if( removedPercentage >= 100 )
      throw new IllegalArgumentException( "RemovedPercentage has to be <100." );
 
    this.removedPercentage = removedPercentage;
  }

  /**
   * Gets the percentage of instances to be removed
   *
   * @return 		the percentage.
   */
  public int getRemovedPercentage() {
    return removedPercentage;
  }

 
  /**
   * Sets the filter used to project the data.
   *
   * @param projectionFilter the filter.
   */
  public void setProjectionFilter( Filter projectionFilter ) {

    this.projectionFilter = projectionFilter;
  }

  /**
   * Gets the filter used to project the data.
   *
   * @return 		the filter.
   */
  public Filter getProjectionFilter() {
    return projectionFilter;
  }

  /**
   * Gets the filter specification string, which contains the class name of
   * the filter and any options to the filter
   *
   * @return the filter string.
   */
  /* Taken from FilteredClassifier */
  protected String getProjectionFilterSpec() {
    
    Filter c = getProjectionFilter();
    if (c instanceof OptionHandler) {
      return c.getClass().getName() + " "
	+ Utils.joinOptions(((OptionHandler)c).getOptions());
    }
    return c.getClass().getName();
  }

    @Override
  public String toString() {
      return "toString not implemented for ContractRotationForest";
  }

  /**
   * builds the classifier.
   *
   * @param data 	the training data to be used for generating the
   * 			classifier.
   * @throws Exception 	if the classifier could not be built successfully
   */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        long startTime=System.currentTimeMillis(); 
        String relationName=data.relationName();
        data = new Instances( data );
        debug=true;
        boolean loadedFromFile=false;
        if(checkpointPath!=null){
            //Look for previous model. Protocol for saving is ProblemName+ClassifierName.ser
            String serPath=checkpointPath+"/"+relationName+"ContractRotationForest.ser";
            System.out.println("Simple name = "+this.getClass().getSimpleName());
            File f =new File(serPath);
            if(f.exists()){
                if(debug)
                    System.out.println("Serialised version exists, trying to load from "+serPath);
                loadFromFile(serPath);
                loadedFromFile=true;
           }
            else
                if(debug)
                    System.out.println("No Serialised versionat location "+serPath);                
        } 
        if(!loadedFromFile){
            if (baseClassifier == null) {
                throw new Exception("A base classifier has not been specified!");
            }
//            m_Classifiers = AbstractClassifier.makeCopies(m_Classifier, m_NumIterations);
            checkMinMax(data);
        //Initialise everything to the max size, then do in batches. 
        //At the end we reduce back to numTrees
            groups=new ArrayList<>();
            // These arrays keep the information of the transformed data set
            headers =new ArrayList<>();
            projectionFilters =new ArrayList<>();
            reducedHeaders = new ArrayList<>();
            classifiers=new ArrayList<>();
        }
        if( data.numInstances() > 0 )// This function fails if there are 0 instances
            random = data.getRandomNumberGenerator(seed);
        else
            random = new Random(seed);


        removeUseless = new RemoveUseless();
        removeUseless.setInputFormat(data);
        data = Filter.useFilter(data, removeUseless);
        normalize = new Normalize();
        normalize.setInputFormat(data);
        data = Filter.useFilter(data, normalize);

        int numClasses = data.numClasses();

        // Split the instances according to their class
        Instances [] instancesOfClass; 
        if( data.classAttribute().isNumeric() ) {
            instancesOfClass = new Instances[numClasses]; 
            instancesOfClass[0] = data;
        }
        else{
            instancesOfClass = new Instances[numClasses+1]; 
            for( int i = 0; i < instancesOfClass.length; i++ ) {
                instancesOfClass[ i ] = new Instances( data, 0 );
            }
            for(Instance instance:data) {
                if( instance.classIsMissing() )
                    instancesOfClass[numClasses].add(instance);
                else{
                    int c = (int)instance.classValue();
                    instancesOfClass[c].add( instance );
                }
            }
          // If there are not instances with a missing class, we do not need to
          // consider them
            if( instancesOfClass[numClasses].numInstances() == 0 ) {
                Instances [] tmp = instancesOfClass;
                instancesOfClass =  new Instances[ numClasses ];
                System.arraycopy( tmp, 0, instancesOfClass, 0, numClasses );
            }
        }


        int n=data.numInstances();
        int m=data.numAttributes()-1;
        double treeTime;
    //Re-estimate even if loading serialised, may be different hardware ....        
        estSingleTree=tm.estimateSingleTreeHours(n,m);
        if(debug){
            System.out.println("n ="+n+" m = "+m+" estSingleTree = "+estSingleTree);
            System.out.println("Contract time ="+contractHours+" hours ");

        }
        int maxAtts=m;
//CASE 1: think we can build the minimum number of trees with full data.
        if((estSingleTree*minNumTrees)<contractHours){ 
            if(debug)
                System.out.println("Think we are able to build at least 50 trees");
            boolean buildFullTree=true;
            int size;
//Option to build in batches for smaller data, but not used at the moment            
            int batchSize=1;//setBatchSize(estSingleTree);    //Set larger for smaller data
//            if(debug)
//                System.out.println("Batch size = "+batchSize);
            long startBuild=System.currentTimeMillis(); 
            while(timeUsed<contractHours && numTrees<maxNumTrees){
                long sTime=System.currentTimeMillis();              
                if(buildFullTree)
                    size=m;
                else{
                    maxAtts=tm.estimateMaxAttributes(m,minNumTrees-numTrees,estSingleTree,contractHours);
                    size=random.nextInt(maxAtts/2)+maxAtts/2;
                }
                    
                if(batchSize+numTrees>maxNumTrees)
                    batchSize=maxNumTrees-numTrees;
                for(int i=0;i<batchSize;i++)
                    buildTreeAttSample(data,instancesOfClass,numTrees++,m);
            //Update time used
                long newTime=System.currentTimeMillis(); 
                timeUsed=(newTime-startBuild)/(1000.0*60.0*60.0);
                treeTime=(newTime-sTime)/(1000.0*60.0*60.0);
                
            //  Update single tree estimate                
                estSingleTree=updateTreeTime(estSingleTree,treeTime,alpha,size,m);
           //Taking much longer than we thought!
                if(estSingleTree*minNumTrees>contractHours)
                    buildFullTree=false;
                else
                    buildFullTree=true;
            //Checkpoint here   
                    if(debug)
                        System.out.println("Built tree number "+numTrees+" in "+timeUsed+" hours ");
                if(checkpointPath!=null){
                    //save the serialised version
                    try{
                        File f=new File(checkpointPath);
                        if(!f.isDirectory())
                            f.mkdirs();
                        saveToFile(checkpointPath+relationName+"ContractRotationForest.ser");
                        if(debug)
                            System.out.println("HERE!!!  Saved to "+checkpointPath+relationName+"ContractRotationForest.ser");
                    }
                    catch(Exception e){
                        System.out.println("Serialisation to "+checkpointPath+"/"+relationName+"ContractRotationForest.ser  FAILED");
                    }
                } 
            }
        }
//CASE 2 and 3: dont think we can build min number of trees        
        else{
            if(debug)
                System.out.println("unable to build 50 trees in the time allowed ");
//If m > n: SAMPLE ATTRIBUTES
            if(m>n){
//estimate maximum number of attributes allowed, x, to get minNumberOfTrees.                
                maxAtts=m;
                long startBuild=System.currentTimeMillis(); 
                while(timeUsed<contractHours && numTrees<minNumTrees){
                    maxAtts=tm.estimateMaxAttributes(m,minNumTrees-numTrees,estSingleTree,contractHours);
                    int size=random.nextInt(maxAtts/2)+maxAtts/2;
                    if(debug){
                        System.out.print("Max estimated attributes ="+maxAtts);
                        System.out.println("    using "+size+" attributes, building single tree at a time. Total time used ="+timeUsed);
                    }
                    long sTime=System.currentTimeMillis();
                    buildTreeAttSample(data,instancesOfClass,numTrees++,size);
            //Update time used
                    long newTime=System.currentTimeMillis(); 
                    timeUsed=(newTime-startBuild)/(1000.0*60.0*60.0);
                    treeTime=(newTime-sTime)/(1000.0*60.0*60.0);
                    estSingleTree=updateTreeTime(estSingleTree,treeTime,alpha,size,m);
//                    (1-alpha)*estSingleTree+alpha*treeTime;
                    if(debug)
                        System.out.println(" actual time used ="+timeUsed+" new est single tree = "+estSingleTree);
                    
                //Checkpoint here   
                }
//Use up any time left here on randomised trees
                while(timeUsed<contractHours && numTrees<maxNumTrees){
                    int size=tm.estimateMaxAttributes(m, 1, estSingleTree,contractHours-timeUsed);
   //                 if(estSingleTree<timeUsed-contractHours || size>m)//Build a whole treee
   //                     size=m;
                    maxAtts*=2;
                    if(maxAtts>size)
                        maxAtts=size;
                    if(debug)
                        System.out.println("OVERTIME: using "+size+" attributes, building single tree at a time. Time used -"+timeUsed);
                    buildTreeAttSample(data,instancesOfClass,numTrees++,maxAtts);
            //Update time used
                    long newTime=System.currentTimeMillis(); 
                    timeUsed=(newTime-startBuild)/(1000.0*60.0*60.0);
                //Checkpoint here   
                    if(debug)
                        System.out.println("Built tree number "+numTrees+" in "+timeUsed+" hours ");
                
                }
            }
            else{
//estimate maximum number of attributes allowed, x, to get minNumberOfTrees.                
                int maxCases=tm.estimateMaxCases(n,minNumTrees,estSingleTree,contractHours);
                if(debug)
                    System.out.println("using max "+maxCases+" case, building single tree at a time");
                long startBuild=System.currentTimeMillis(); 
                while(timeUsed<contractHours && numTrees<minNumTrees){
                    int size=random.nextInt(maxCases/2)+maxCases/2;
                    buildTreeCaseSample(data,instancesOfClass,numTrees++,size);
            //Update time used
                    long newTime=System.currentTimeMillis(); 
                    timeUsed=(newTime-startBuild)/(1000.0*60.0*60.0);
                //Checkpoint here   
                }
//Use up any time left here on randomised trees
                while(timeUsed<contractHours && numTrees<maxNumTrees){
                    int size=tm.estimateMaxCases(n, 1, estSingleTree,contractHours-timeUsed);
                    buildTreeCaseSample(data,instancesOfClass,numTrees++,size);
            //Update time used
                    long newTime=System.currentTimeMillis(); 
                    timeUsed=(newTime-startBuild)/(1000.0*60.0*60.0);
                //Checkpoint here   
                    if(debug)
                        System.out.println("Built tree number "+numTrees+" in "+timeUsed+" hours ");
                
                }
            }
        }
        res.buildTime=System.currentTimeMillis()-startTime;
        if(debug)
            System.out.println("Finished build");

    }
    double updateTreeTime(double estSingleTree,double obsTreeTime,double alpha,int numAtts,int m){
        double t=(1-alpha)*estSingleTree;
        t+=alpha*(m/(double)numAtts)*obsTreeTime;
        if(t<0)
            return estSingleTree;
        return t;
    }

    
/** Build a rotation forest tree on a random subsample of the attributes
 * 
 * @param data
 * @param instancesOfClass
 * @param i
 * @param numAtts
 * @throws Exception 
 */    
 public void buildTreeAttSample(Instances data, Instances [] instancesOfClass,int i, int numAtts) throws Exception{
        int[][] g=generateGroupFromSize(data, random,numAtts);
        Filter[] projection=Filter.makeCopies(projectionFilter, g.length );
        projectionFilters.add(projection);
        groups.add(g);
        Instances[] reducedHeaders = new Instances[ g.length ];
        this.reducedHeaders.add(reducedHeaders);

        FastVector transformedAttributes = new FastVector( data.numAttributes() );
        // Construction of the dataset for each group of attributes
        for( int j = 0; j < g.length; j++ ) {
            FastVector fv = new FastVector( g[j].length + 1 );
            for( int k = 0; k < g[j].length; k++ ) {
              String newName = data.attribute( g[j][k] ).name()
                + "_" + k;
              fv.addElement( data.attribute( g[j][k] ).copy(newName) );
            }
            fv.addElement( data.classAttribute( ).copy() );
            Instances dataSubSet = new Instances( "rotated-" + i + "-" + j + "-", 
                fv, 0);
            dataSubSet.setClassIndex( dataSubSet.numAttributes() - 1 );
            // Select instances for the dataset
            reducedHeaders[j] = new Instances( dataSubSet, 0 );
            boolean [] selectedClasses = selectClasses( instancesOfClass.length, 
                  random );
            for( int c = 0; c < selectedClasses.length; c++ ) {
                if( !selectedClasses[c] )
                    continue;
                for(Instance instance:instancesOfClass[c]) {
                    Instance newInstance = new DenseInstance(dataSubSet.numAttributes());
                    newInstance.setDataset( dataSubSet );
                    for( int k = 0; k < g[j].length; k++ ) {
                      newInstance.setValue( k, instance.value( g[j][k] ) );
                    }
                    newInstance.setClassValue( instance.classValue( ) );
                    dataSubSet.add( newInstance );
                }
            }
            dataSubSet.randomize(random);
            // Remove a percentage of the instances
            Instances originalDataSubSet = dataSubSet;
            dataSubSet.randomize(random);
            RemovePercentage rp = new RemovePercentage();
            rp.setPercentage(removedPercentage );
            rp.setInputFormat( dataSubSet );
            dataSubSet = Filter.useFilter( dataSubSet, rp );
            if( dataSubSet.numInstances() < 2 ) {
                dataSubSet = originalDataSubSet;
            }
        // Project the data
        
            projection[j].setInputFormat( dataSubSet );
            Instances projectedData = null;
            do {
                try {
                    projectedData = Filter.useFilter( dataSubSet, 
                    projection[j] );
                } catch ( Exception e ) {
                // The data could not be projected, we add some random instances
                    addRandomInstances( dataSubSet, 10, random );
                }
            } while( projectedData == null );

            // Include the projected attributes in the attributes of the 
            // transformed dataset
            for( int a = 0; a < projectedData.numAttributes() - 1; a++ ) {
                String newName = projectedData.attribute(a).name() + "_" + j;
                transformedAttributes.addElement( projectedData.attribute(a).copy(newName));
            }
        }
      
        transformedAttributes.addElement( data.classAttribute().copy() );
        Instances buildClas = new Instances( "rotated-" + i + "-", 
            transformedAttributes, 0 );
        buildClas.setClassIndex( buildClas.numAttributes() - 1 );
        headers.add(new Instances( buildClas, 0 ));

      // Project all the training data
        for(Instance instance:data) {
            Instance newInstance = convertInstance( instance, i );
            buildClas.add( newInstance );
        }
        Classifier c= AbstractClassifier.makeCopy(baseClassifier);
        // Build the base classifier
        if (c instanceof Randomizable) {
            ((Randomizable) c).setSeed(random.nextInt());
        }
        c.buildClassifier( buildClas );
        classifiers.add(c);
    }
 
/** Build a rotation forest tree on a random subsample of the instances
 * 
 * @param data
 * @param instancesOfClass
 * @param i
 * @param numCases
 * @throws Exception 
 */
  public void buildTreeCaseSample(Instances data, Instances [] instancesOfClass,int i, int numCases) throws Exception{
        int[][] g=generateGroupFromSize(data, random,data.numAttributes()-1);
        Filter[] projection=Filter.makeCopies(projectionFilter, g.length );
        projectionFilters.add(projection);
        groups.add(g);
        Instances[] reducedHeaders = new Instances[ g.length ];
        this.reducedHeaders.add(reducedHeaders);
        data=new Instances(data);
        int m=data.numInstances();
        for(int k=0;k<m-numCases;k++)
            data.remove(random.nextInt(data.numInstances()));
        
        
        FastVector transformedAttributes = new FastVector( data.numAttributes() );
        // Construction of the dataset for each group of attributes
        for( int j = 0; j < g.length; j++ ) {
            FastVector fv = new FastVector( g[j].length + 1 );
            for( int k = 0; k < g[j].length; k++ ) {
              String newName = data.attribute( g[j][k] ).name()
                + "_" + k;
              fv.addElement( data.attribute( g[j][k] ).copy(newName) );
            }
            fv.addElement( data.classAttribute( ).copy() );
            Instances dataSubSet = new Instances( "rotated-" + i + "-" + j + "-", 
                fv, 0);
            dataSubSet.setClassIndex( dataSubSet.numAttributes() - 1 );
            // Select instances for the dataset
            reducedHeaders[j] = new Instances( dataSubSet, 0 );
            boolean [] selectedClasses = selectClasses( instancesOfClass.length, 
                  random );
            for( int c = 0; c < selectedClasses.length; c++ ) {
                if( !selectedClasses[c] )
                    continue;
                for(Instance instance:instancesOfClass[c]) {
                    Instance newInstance = new DenseInstance(dataSubSet.numAttributes());
                    newInstance.setDataset( dataSubSet );
                    for( int k = 0; k < g[j].length; k++ ) {
                      newInstance.setValue( k, instance.value( g[j][k] ) );
                    }
                    newInstance.setClassValue( instance.classValue( ) );
                    dataSubSet.add( newInstance );
                }
            }
            dataSubSet.randomize(random);
            // Remove a percentage of the instances
            Instances originalDataSubSet = dataSubSet;
            dataSubSet.randomize(random);
            RemovePercentage rp = new RemovePercentage();
            rp.setPercentage(removedPercentage );
            rp.setInputFormat( dataSubSet );
            dataSubSet = Filter.useFilter( dataSubSet, rp );
            if( dataSubSet.numInstances() < 2 ) {
                dataSubSet = originalDataSubSet;
            }
        // Project the data
        
            projection[j].setInputFormat( dataSubSet );
            Instances projectedData = null;
            do {
                try {
                    projectedData = Filter.useFilter( dataSubSet, 
                    projection[j] );
                } catch ( Exception e ) {
                // The data could not be projected, we add some random instances
                    addRandomInstances( dataSubSet, 10, random );
                }
            } while( projectedData == null );

            // Include the projected attributes in the attributes of the 
            // transformed dataset
            for( int a = 0; a < projectedData.numAttributes() - 1; a++ ) {
                String newName = projectedData.attribute(a).name() + "_" + j;
                transformedAttributes.addElement( projectedData.attribute(a).copy(newName));
            }
        }
      
        transformedAttributes.addElement( data.classAttribute().copy() );
        Instances buildClas = new Instances( "rotated-" + i + "-", 
            transformedAttributes, 0 );
        buildClas.setClassIndex( buildClas.numAttributes() - 1 );
        headers.add(new Instances( buildClas, 0 ));

      // Project all the training data
        for(Instance instance:data) {
            Instance newInstance = convertInstance( instance, i );
            buildClas.add( newInstance );
        }
        Classifier c= AbstractClassifier.makeCopy(baseClassifier);
        // Build the base classifier
        if (c instanceof Randomizable) {
            ((Randomizable) c).setSeed(random.nextInt());
        }
        c.buildClassifier( buildClas );
        classifiers.add(c);
    }
 

 
 private int setBatchSize(double singleTreeHours){
        if(singleTreeHours>ContractClassifier.CHECKPOINTINTERVAL)
            return 1;
        int hrs=(int)(ContractClassifier.CHECKPOINTINTERVAL/singleTreeHours);
        return hrs;
        
    }

 
  /** 
   * Adds random instances to the dataset.
   * 
   * @param dataset the dataset
   * @param numInstances the number of instances
   * @param random a random number generator
   */
  protected void addRandomInstances( Instances dataset, int numInstances, 
                                  Random random ) {
    int n = dataset.numAttributes();				
    double [] v = new double[ n ];
    for( int i = 0; i < numInstances; i++ ) {
      for( int j = 0; j < n; j++ ) {
        Attribute att = dataset.attribute( j );
        if( att.isNumeric() ) {
	  v[ j ] = random.nextDouble();
	}
	else if ( att.isNominal() ) { 
	  v[ j ] = random.nextInt( att.numValues() );
	}
      }
      dataset.add( new DenseInstance( 1, v ) );
    }
  }

  /** 
   * Checks minGroup and maxGroup
   * 
   * @param data the dataset
   */
  protected void checkMinMax(Instances data) {
    if( minGroup > maxGroup ) {
      int tmp = maxGroup;
      maxGroup = minGroup;
      minGroup = tmp;
    }
    
    int n = data.numAttributes();
    if( maxGroup >= n )
      maxGroup = n - 1;
    if( minGroup >= n )
      minGroup = n - 1;
  }

  /** 
   * Selects a non-empty subset of the classes
   * 
   * @param numClasses         the number of classes
   * @param random 	       the random number generator.
   * @return a random subset of classes
   */
  protected boolean [] selectClasses( int numClasses, Random random ) {

    int numSelected = 0;
    boolean selected[] = new boolean[ numClasses ];

    for( int i = 0; i < selected.length; i++ ) {
      if(random.nextBoolean()) {
        selected[i] = true;
        numSelected++;
      }
    }
    if( numSelected == 0 ) {
      selected[random.nextInt( selected.length )] = true;
    }
    return selected;
  }

  
/**
   * generates the groups of attributes, given their minimum and maximum
   * sizes.
   *
   * @param data 	the training data to be used for generating the
   * 			groups.
   * @param random 	the random number generator.
   */
  protected int[][] generateGroupFromSize(Instances data, Random random, int maxAtts) {
    int[][] groups;
    int [] permutation = attributesPermutation(data.numAttributes(), 
                           data.classIndex(), random, maxAtts);

      // The number of groups that have a given size 
      int [] numGroupsOfSize = new int[maxGroup - minGroup + 1];

      int numAttributes = 0;
      int numGroups;

      // Select the size of each group
      for( numGroups = 0; numAttributes < permutation.length; numGroups++ ) {
        int n = random.nextInt( numGroupsOfSize.length );
        numGroupsOfSize[n]++;
        numAttributes += minGroup + n;
      }

      groups = new int[numGroups][];
      int currentAttribute = 0;
      int currentSize = 0;
      for( int j = 0; j < numGroups; j++ ) {
        while( numGroupsOfSize[ currentSize ] == 0 )
          currentSize++;
        numGroupsOfSize[ currentSize ]--;
        int n = minGroup + currentSize;
        groups[j] = new int[n];
        for( int k = 0; k < n; k++ ) {
          if( currentAttribute < permutation.length )
            groups[j][k] = permutation[ currentAttribute ];
          else
	    // For the last group, it can be necessary to reuse some attributes
            groups[j][k] = permutation[ random.nextInt( 
	        permutation.length ) ];
          currentAttribute++;
        }
      }
      return groups;
  }  
  
 

     final protected int [] attributesPermutation(int numAttributes, int classAttribute,
                                         Random random, int maxNumAttributes) {
        int [] permutation = new int[numAttributes-1];
        int i = 0;
        //This just ignores the class attribute
        for(; i < classAttribute; i++){
          permutation[i] = i;
        }
        for(; i < permutation.length; i++){
          permutation[i] = i + 1;
        }

        permute( permutation, random );
        if(numAttributes>maxNumAttributes){
        //TRUNCTATE THE PERMATION TO CONSIDER maxNumAttributes. 
        // we could do this more efficiently, but this is the simplest way. 
            int[] temp = new int[maxNumAttributes];
           System.arraycopy(permutation, 0, temp, 0, maxNumAttributes);
           permutation=temp;
        }
    return permutation;
    }    

  /**
   * permutes the elements of a given array.
   *
   * @param v       the array to permute
   * @param random  the random number generator.
   */
  protected void permute( int v[], Random random ) {

    for(int i = v.length - 1; i > 0; i-- ) {
      int j = random.nextInt( i + 1 );
      if( i != j ) {
        int tmp = v[i];
        v[i] = v[j];
        v[j] = tmp;
      }
    }
  }

  /**
   * prints the groups.
   */
  protected void printGroups( ) {
    for( int i = 0; i < groups.size(); i++ ) {
      for( int j = 0; j < groups.get(i).length; j++ ) {
        System.err.print( "( " );
        for( int k = 0; k < groups.get(i)[j].length; k++ ) {
          System.err.print(groups.get(i)[j][k] );
          System.err.print( " " );
        }
        System.err.print( ") " );
      }
      System.err.println( );
    }
  }

  /** 
   * Transforms an instance for the i-th classifier.
   *
   * @param instance the instance to be transformed
   * @param i the base classifier number
   * @return the transformed instance
   * @throws Exception if the instance can't be converted successfully 
   */
  protected Instance convertInstance( Instance instance, int i ) 
  throws Exception {
    Instance newInstance = new DenseInstance( headers.get(i).numAttributes( ) );
    newInstance.setWeight(instance.weight());
    newInstance.setDataset(headers.get(i));
    int currentAttribute = 0;

    // Project the data for each group
    int[][] g=groups.get(i);
    for( int j = 0; j < g.length; j++ ) {
      Instance auxInstance = new DenseInstance(g[j].length + 1 );
      int k;
      for( k = 0; k < g[j].length; k++ ) {
        auxInstance.setValue( k, instance.value( g[j][k] ) );
      }
      auxInstance.setValue( k, instance.classValue( ) );
      auxInstance.setDataset(reducedHeaders.get(i)[ j ] );
      Filter[] projection=projectionFilters.get(i);
      projection[j].input( auxInstance );
      auxInstance = projection[j].output( );
      projection[j].batchFinished();
      for( int a = 0; a < auxInstance.numAttributes() - 1; a++ ) {
        newInstance.setValue( currentAttribute++, auxInstance.value( a ) );
      }
    }

    newInstance.setClassValue( instance.classValue() );
    return newInstance;
  }

  /**
   * Calculates the class membership probabilities for the given test
   * instance.
   *
   * @param instance the instance to be classified
   * @return preedicted class probability distribution
   * @throws Exception if distribution can't be computed successfully 
   */
  @Override
  public double[] distributionForInstance(Instance instance) throws Exception {

    removeUseless.input(instance);
    instance =removeUseless.output();
    removeUseless.batchFinished();

    normalize.input(instance);
    instance =normalize.output();
    normalize.batchFinished();

    double [] sums = new double [instance.numClasses()], newProbs; 
    
    for (int i = 0; i < classifiers.size(); i++) {
      Instance convertedInstance = convertInstance(instance, i);
      if (instance.classAttribute().isNumeric() == true) {
	sums[0] += classifiers.get(i).classifyInstance(convertedInstance);
      } else {
	newProbs = classifiers.get(i).distributionForInstance(convertedInstance);
	for (int j = 0; j < newProbs.length; j++)
	  sums[j] += newProbs[j];
      }
    }
    if (instance.classAttribute().isNumeric() == true) {
      sums[0] /= (double)classifiers.size();
      return sums;
    } else if (Utils.eq(Utils.sum(sums), 0)) {
      return sums;
    } else {
      Utils.normalize(sums);
      return sums;
    }
  }

    @Override
    public String getParameters() {
        String result="BuildTime,"+res.buildTime+",CVAcc,"+res.acc+",RemovePercent,"+this.getRemovedPercentage()+",NumFeatures,"+this.getMaxGroup();
        result+=",numTrees,"+numTrees;
        return result;
    }
    @Override
    public void setTimeLimit(long time) {
        contractHours=((double)time)/(1000.0*60.0*60);
    }

    @Override
    public void setTimeLimit(TimeLimit time, int amount) {
        switch(time){
            case MINUTE:
                contractHours=((double)amount)/60.0;
                break;
            case DAY:
                contractHours=((double)amount*24.0);
                break;
            case HOUR: default:
                contractHours=((double)amount);
                break;
        }
    }

    @Override
    public void setSavePath(String path) {
        checkpointPath=path;
    }

    @Override
    public void copyFromSerObject(Object obj) throws Exception {
        if(!(obj instanceof ContractRotationForest))
            throw new Exception("The SER file is not am instance of ContractRotationForest"); //To change body of generated methods, choose Tools | Templates.
        ContractRotationForest saved= ((ContractRotationForest)obj);

//Copy RotationForest attributes
        baseClassifier=saved.baseClassifier;
        classifiers=saved.classifiers;
        minGroup = saved.minGroup;
        maxGroup = saved.maxGroup;
        removedPercentage = saved.removedPercentage;
        groups = saved.groups;
        projectionFilter = saved.projectionFilter;
        projectionFilters = saved.projectionFilters;
        headers = saved.headers;
        reducedHeaders = saved.reducedHeaders;
        removeUseless = saved.removeUseless;
        normalize = saved.normalize;

  
//Copy ContractRotationForest attributes
        this.contractHours=saved.contractHours;
        res=saved.res;
        minNumTrees=saved.minNumTrees;
        maxNumTrees=saved.maxNumTrees;
        maxNumAttributes=saved.maxNumAttributes;
        checkpointPath=saved.checkpointPath;
        debug=saved.debug;
        tm=saved.tm;
        timeUsed=saved.timeUsed;
        numTrees=saved.numTrees;
    
    }
  
  /**
   * Main method for testing this class.
   *
   * @param argv the options
   */
  public static void main(String [] argv) throws Exception {
      ContractRotationForest cf =new ContractRotationForest();
      Class cls=cf.getClass();
      System.out.println("Class canonical name ="+cls.getCanonicalName()+" class simple name "+cls.getSimpleName()+" class full name ="+cls.getName());
      String path="C:/temp/ItalyPowerDemandContractRotationForest.ser";
        FileInputStream fis = new FileInputStream(path);
        ObjectInputStream in = new ObjectInputStream(fis);
        Object crf =in.readObject();
        in.close();
        TimingModel tm=cf.new TimingModel();

  }
   
    private class TimingModel implements Serializable{
        double b0,b1,b2,b3,b4;
        double predictionInterval=3.67;
        double baseNumberOfTrees=200;
//Time taken to do a standard operation on the model build computer        
        static final double BASEFACTOR=1;
        double normalisingFactor;
        public TimingModel(){
//Default model an+bm+cmn    






            b0=0.679693678;
            b1=0.000132076; //n
            b2=0.000245885;//m
            b3=1.23057E-06;//mn
            normalisingFactor=normalise();
        }
        double estimateSingleTreeHours(int n, int m){
//Estimate time    
            double t=b0+b1*n+b2*m+b3*n*m+predictionInterval;
            t*=normalisingFactor/BASEFACTOR;
            t/=baseNumberOfTrees;
//Normalise for this computer
            return t; //This is a fraction of an hour! so .1 ==6 minutes
        }
        final double  normalise(){
            return 1.0;
        }
//estimate of the number of possible attributes to build numTrees given a contract time         
        int estimateMaxAttributes(int m, int numTrees, double singleTreeTime, double contractTime){
            
            double estM=(m*contractTime)/(numTrees*singleTreeTime);
            if(estM<3)
                estM=3;
            else if(estM>m)
                estM=m;
            return (int)(estM);
        }
        int estimateMaxCases(int n, int numTrees, double singleTreeTime, double contractTime){
            double estN=(n*contractTime)/(numTrees*singleTreeTime);
            if(estN<3)
                estN=3;
            else if(estN>n)
                estN=n;
            return (int)(estN);
        }


    }
 
}

