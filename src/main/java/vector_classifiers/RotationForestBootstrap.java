/*
 Adjusted Rotation Forest.

VERSION 1: 


Version 2. Impose bootstrapping and work out OOB Error

Rotation forest samples 50% of the cases for each attribute set of any given tree. 
This means you cannot find OOB and need to CV to get an error estimate.
This class simply does the sampling for each tree rather than each set. 
To Do:
1. Implement the sampling
2. Test whether it makes a difference to test acc
3. Implement the calculation out of bag error and probabilities and debug
4. Test difference in CV train and OOB error estimates

cant really wrap this easily using the bagging clas, so will have to implement the bagging. 
1. Set m_RemovedPercentage=0 

 */
package vector_classifiers;

import fileIO.OutFile;
import java.util.Enumeration;
import java.util.Random;
import utilities.CrossValidator;
import vector_classifiers.TunedRandomForest.EnhancedBagging;
import weka.classifiers.meta.RotationForest;
import vector_classifiers.TunedRotationForest;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 *
 * @author ajb
 */
public class RotationForestBootstrap extends TunedRotationForest{
    ///        boolean[][] inBags;
    double trainPercentPerTree=0.5;
    
    public RotationForestBootstrap(){
        super();
        m_RemovedPercentage=0;
//This is not going to work.         
//        EnhancedBagging base=new EnhancedBagging();
//        base.setClassifier(new weka.classifiers.trees.J48());
//        m_Classifier=base;
        m_ProjectionFilter = defaultFilter();

    }
/*    
    @Override
    protected int [] attributesPermutation(int numAttributes, int classAttribute,
                                         Random random) {

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
 */
/**
   * builds the classifier.
   *
   * @param data 	the training data to be used for generating the
   * 			classifier.
   * @throws Exception 	if the classifier could not be built successfully
   */
    private Instances[] splitByClass(Instances data){
        int numClasses=data.numClasses();
        Instances [] instancesOfClass = new Instances[numClasses + 1]; 
        if( data.classAttribute().isNumeric() ) {
          instancesOfClass = new Instances[numClasses]; 
          instancesOfClass[0] = data;
        }
        else {
          instancesOfClass = new Instances[numClasses+1]; 
          for( int i = 0; i < instancesOfClass.length; i++ ) {
            instancesOfClass[ i ] = new Instances( data, 0 );
          }
          Enumeration enu = data.enumerateInstances();
          while( enu.hasMoreElements() ) {
            Instance instance = (Instance)enu.nextElement();
            if( instance.classIsMissing() ) {
              instancesOfClass[numClasses].add( instance );
            }
            else {
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
        return instancesOfClass;
    }
 
     @Override
    public void buildClassifier(Instances data) throws Exception{
//        res.buildTime=System.currentTimeMillis(); //removed with cv changes  (jamesl) 
        long startTime=System.currentTimeMillis(); 
        //now calced separately from any instance on ClassifierResults, and added on at the end

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    data = new Instances( data );
    super.buildClassifier(data);

    checkMinMax(data);

    Random random;
    if( data.numInstances() > 0 ) {
      // This function fails if there are 0 instances
      random = data.getRandomNumberGenerator(m_Seed);
    }
    else {
      random = new Random(m_Seed);
    }

    m_RemoveUseless = new RemoveUseless();
    m_RemoveUseless.setInputFormat(data);
    data = Filter.useFilter(data, m_RemoveUseless);

    m_Normalize = new Normalize();
    m_Normalize.setInputFormat(data);
    data = Filter.useFilter(data, m_Normalize);

    if(m_NumberOfGroups) {
      generateGroupsFromNumbers(data, random);
    }
    else {
      generateGroupsFromSizes(data, random);
    }

    m_ProjectionFilters = new Filter[m_Groups.length][];
    for(int i = 0; i < m_ProjectionFilters.length; i++ ) {
      m_ProjectionFilters[i] = Filter.makeCopies( m_ProjectionFilter, 
          m_Groups[i].length );
    }


    // These arrays keep the information of the transformed data set
    m_Headers = new Instances[ m_Classifiers.length ];
    m_ReducedHeaders = new Instances[ m_Classifiers.length ][];

    // Construction of the base classifiers
    for(int i = 0; i < m_Classifiers.length; i++) {
        int numClasses = data.numClasses();

//HERE THE ONLY DIFFERENT BIT
        Instances trainData=new Instances(data);
        Instances testData=new Instances(data,0);
        trainData.randomize(rng);
        
        int trainSize=(int)(trainPercentPerTree*data.numInstances());
        while(trainSize<trainData.numInstances())
            testData.add(trainData.remove(0));
//MOVE THIS BIT BECAUSE THE DATA IS DIFFERENT NOW PER TREE
// Split the instances according to their class: 
//Doing unnecessary work in the spirit of simplicity of code
/*Easy to optimise to do the split once, or indeed, see if it makes any difference
        */        
        Instances [] instancesOfClass = splitByClass(trainData); 
        m_ReducedHeaders[i] = new Instances[ m_Groups[i].length ];
      FastVector transformedAttributes = new FastVector( trainData.numAttributes() );

        // Construction of the dataset for each group of attributes
        for( int j = 0; j < m_Groups[ i ].length; j++ ) {
        FastVector fv = new FastVector( m_Groups[i][j].length + 1 );
        for( int k = 0; k < m_Groups[i][j].length; k++ ) {
          String newName = trainData.attribute( m_Groups[i][j][k] ).name()
            + "_" + k;
          fv.addElement( trainData.attribute( m_Groups[i][j][k] ).copy(newName) );
        }
        fv.addElement( trainData.classAttribute( ).copy() );
        Instances dataSubSet = new Instances( "rotated-" + i + "-" + j + "-", 
            fv, 0);
        dataSubSet.setClassIndex( dataSubSet.numAttributes() - 1 );
        // Select instances for the dataset
        m_ReducedHeaders[i][j] = new Instances( dataSubSet, 0 );
        boolean [] selectedClasses = selectClasses( instancesOfClass.length, 
	      random );
        for( int c = 0; c < selectedClasses.length; c++ ) {
          if( !selectedClasses[c] )
            continue;
          Enumeration enu = instancesOfClass[c].enumerateInstances();
          while( enu.hasMoreElements() ) {
            Instance instance = (Instance)enu.nextElement();
            Instance newInstance = new DenseInstance(dataSubSet.numAttributes());
            newInstance.setDataset( dataSubSet );
            for( int k = 0; k < m_Groups[i][j].length; k++ ) {
              newInstance.setValue( k, instance.value( m_Groups[i][j][k] ) );
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
        rp.setPercentage( m_RemovedPercentage );
        rp.setInputFormat( dataSubSet );
        dataSubSet = Filter.useFilter( dataSubSet, rp );
	if( dataSubSet.numInstances() < 2 ) {
	  dataSubSet = originalDataSubSet;
	}

        // Project de data
        m_ProjectionFilters[i][j].setInputFormat( dataSubSet );
	Instances projectedData = null;
	do {
	  try {
            projectedData = Filter.useFilter( dataSubSet, 
	        m_ProjectionFilters[i][j] );
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
      
      transformedAttributes.addElement( trainData.classAttribute().copy() );
      Instances buildClas = new Instances( "rotated-" + i + "-", 
        transformedAttributes, 0 );
      buildClas.setClassIndex( buildClas.numAttributes() - 1 );
      m_Headers[ i ] = new Instances( buildClas, 0 );

      // Project all the training data
      Enumeration enu = trainData.enumerateInstances();
      while( enu.hasMoreElements() ) {
        Instance instance = (Instance)enu.nextElement();
        Instance newInstance = convertInstance( instance, i );
        buildClas.add( newInstance );
      }

      // Build the base classifier
      if (m_Classifier instanceof Randomizable) {
	((Randomizable) m_Classifiers[i]).setSeed(random.nextInt());
      }
      m_Classifiers[i].buildClassifier( buildClas );
    }

    if(m_Debug){
      printGroups();
    }        res.buildTime=System.currentTimeMillis()-startTime;
    }
  
      
    
//Bagging    
    public static void main(String[] args){
        
    }
}
