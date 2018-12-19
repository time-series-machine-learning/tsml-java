/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package vector_classifiers;

import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 *
 * @author cjr13geu
 */
public class RotationTree extends AbstractClassifier{
    
    private Classifier baseClassifier = null;
    private Filter[] projectionFilters = null;
    private Filter projectionFilter = null;
    private int minGroupSize;
    private int maxGroupSize;
    private RemoveUseless removeUseless = null;
    private Normalize normalise = null;
    private int groups[][] = null;
    private int seed;
    private Instances attributeDetails = null;
    private Instances[] reducedHeaders = null;
    private int removePercentage;
    
    public RotationTree(){  
        initialise();
    }
    
    public RotationTree(int seed){
        this.seed = seed;
        initialise();
    }
    
    public void setSeed(int x){
        this.seed = x;
    }
    
    private void initialise(){
        removePercentage = 0;
        minGroupSize = 3;
        maxGroupSize = 3;
        baseClassifier = new J48();
        projectionFilter = defaultFilter();
        removeUseless = new RemoveUseless();
        normalise = new Normalize();
    }
    
    public void removePercentage(int percentage){
        removePercentage = percentage;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        
        
        Random random;
        if( data.numInstances() > 0 ) {
          // This function fails if there are 0 instances
          random = data.getRandomNumberGenerator(seed);
        }
        else {
          random = new Random(seed);
        }
        
        checkGroupSizesValid(data);
        removeUseless.setInputFormat(data);
        data = Filter.useFilter(data, removeUseless);
        normalise.setInputFormat(data);
        data = Filter.useFilter(data, normalise);
        generateGroupsFromNumAtts(data, random);
        projectionFilters = new Filter[groups.length];
        reducedHeaders = new Instances[groups.length];
        
        for (int i = 0; i < groups.length; i++) {
            projectionFilters[i] = Filter.makeCopy(projectionFilter);
        }
        
        //Split data into instances[] based on class;
        Instances[] dataSplitByClass = splitByClass(data);
        ArrayList<Attribute> transformedAttributes = new ArrayList<>();
        
        
        for (int i = 0; i < groups.length; i++) {
            ArrayList<Attribute> temp = new ArrayList<>();
            
            for (int j = 0; j < groups[i].length; j++) {
                String newName = data.attribute(groups[i][j]).name() + "_" + j;
                temp.add(data.attribute(groups[i][j]).copy(newName));
            }
            temp.add((Attribute)data.classAttribute().copy());
            Instances dataSubSet = new Instances("Rotated-" + seed + "-" + i + "-", temp, 0);
            dataSubSet.setClassIndex(dataSubSet.numAttributes() - 1);
            
            //Select Instances to be used in creating the PCA for this group.
            reducedHeaders[i] = new Instances(dataSubSet, 0);
            boolean[] selectedClasses = selectClasses(dataSplitByClass.length, random);
            for (int j = 0; j < selectedClasses.length; j++) {
                if(!selectedClasses[j]){
                    continue;
                }
                for (int k = 0; k < dataSplitByClass[j].size(); k++) {
                    Instance newInstance = new DenseInstance(dataSubSet.numAttributes());
                    newInstance.setDataset(dataSubSet);
                    for (int l = 0; l < groups[i].length; l++) {
                        newInstance.setValue(l, dataSplitByClass[j].get(k).value(groups[i][l]));
                    }
                    newInstance.setClassValue(dataSplitByClass[j].get(k).classValue());
                    dataSubSet.add(newInstance);
                }
            }
            
            dataSubSet.randomize(random);
            Instances originalDataSubSet = dataSubSet;
            dataSubSet.randomize(random);
            RemovePercentage rp = new RemovePercentage();
            rp.setPercentage(removePercentage);
            rp.setInputFormat( dataSubSet );
            dataSubSet = Filter.useFilter( dataSubSet, rp );
            if( dataSubSet.numInstances() < 2 ) {
              dataSubSet = originalDataSubSet;
            }
            
            //Project the data.
            projectionFilters[i].setInputFormat(dataSubSet);
            Instances projectedData = null;
            do{
                try{
                    projectedData = Filter.useFilter(dataSubSet, projectionFilters[i]);
                }catch(Exception e){
                    addRandomInstances(dataSubSet, 10, random);
                }
            }while(projectedData == null);
            
            //Append the projected attributes to the transformedAttrbutes arrayList.
            for (int j = 0; j < projectedData.numAttributes() - 1; j++) {
                String newName = projectedData.attribute(j).name() + "_" + i;
                transformedAttributes.add(projectedData.attribute(j).copy(newName));
            }
        }
        
        transformedAttributes.add((Attribute)data.classAttribute().copy());
        Instances buildClass;
        buildClass = new Instances("Rotated-" + seed + "-", transformedAttributes, 0); 
        buildClass.setClassIndex(buildClass.numAttributes() - 1);
        attributeDetails = new Instances(buildClass, 0);
        
        //Project all training data.
        for (int i = 0; i < data.size(); i++) {
            Instance temp = convertInstance(data.get(i));
            buildClass.add(temp);
        }
        baseClassifier.buildClassifier(buildClass);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[]distribution = distributionForInstance(instance);

        int maxVote=0;
        for(int i = 1; i < distribution.length; i++)
            if(distribution[i] > distribution[maxVote])
                maxVote = i;
        return maxVote;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        
        removeUseless.input(instance);
        instance = removeUseless.output();
        removeUseless.batchFinished();
        
        normalise.input(instance);
        instance = normalise.output();
        removeUseless.batchFinished();
        
        double sums[] = new double[instance.numClasses()];
        double newProbs[];
        
        Instance convertedInstance = convertInstance(instance);
        if(instance.classAttribute().isNumeric()){
            sums[0] += baseClassifier.classifyInstance(convertedInstance);
        }else{
            sums = baseClassifier.distributionForInstance(convertedInstance);   
        }
        return sums;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    protected Filter defaultFilter() {
        PrincipalComponents filter = new PrincipalComponents();
        //filter.setNormalize(false);
        filter.setVarianceCovered(1.0);
        return filter;
    }
    
    public void setMinGroupSize(int x){
        minGroupSize = x;
    }
    
    public void setMaxGroupSize(int x){
        maxGroupSize = x;
    }
    
    private void checkGroupSizesValid(Instances data){
        if( minGroupSize > maxGroupSize ) {
            int tmp = maxGroupSize;
            maxGroupSize = minGroupSize;
            minGroupSize = tmp;
        }

      int n = data.numAttributes();
      if( maxGroupSize >= n )
        maxGroupSize = n - 1;
      if( minGroupSize >= n )
        minGroupSize = n - 1;
    }
    
    private void generateGroupsFromNumAtts(Instances data, Random random){
        
        //Returns index of attributes in random order.
        int[] permutation = attributesPermutation(data.numAttributes(), data.classIndex(), random);
        //Number of groups with given size (minGroupSize + numGroupsOfSize[x]). Where,
        //number of groups of size minGroupSize = minGroupSize + numGroupsSize[0].
        int[] numGroupsOfSize = new int[maxGroupSize - minGroupSize + 1];
        int numAttributes  = 0;
        int numGroups;
        
        for (numGroups = 0; numAttributes < permutation.length; numGroups++) {
            int n = random.nextInt(numGroupsOfSize.length);
            numGroupsOfSize[n]++;
            numAttributes += minGroupSize + n;
        }
        
        groups = new int[numGroups][];
        int currentAttribute = 0;
        int currentSize = 0;
        
        for (int i = 0; i < numGroups; i++) {
            
            while(numGroupsOfSize[currentSize] == 0)
                currentSize++;
            
            numGroupsOfSize[currentSize]--;
            int groupSize = minGroupSize + currentSize;
            groups[i] = new int[groupSize];
            
            for (int j = 0; j < groupSize; j++) {
                if (currentAttribute < permutation.length) {
                    groups[i][j] = permutation[currentAttribute];
                }else{
                    //For the last group there may be duplicate attributes.
                    groups[i][j] = permutation[ random.nextInt(permutation.length)];     
                }
                currentAttribute++;
            }
        }    
    }
    
    /**
    * generates a permutation of the attributes.
    *
    * @param numAttributes       the number of attributes.
    * @param classAttributes     the index of the class attribute.
    * @param random 	        the random number generator.
    * @return a permutation of the attributes
    */
    protected int [] attributesPermutation(int numAttributes, int classAttribute, Random random){
        int [] permutation = new int[numAttributes-1];
        int i = 0;
        
        for(; i < classAttribute; i++){
            permutation[i] = i;
        }
        for(; i < permutation.length; i++){
            permutation[i] = i + 1;
        }
        
        permute( permutation, random);
        return permutation;
    }
    
    /**
    * permutes the elements of a given array.
    *
    * @param v       the array to permute
    * @param random  the random number generator.
    */
    protected void permute( int v[], Random random){

        for(int i = v.length - 1; i > 0; i-- ){
            int j = random.nextInt( i + 1 );
            if( i != j ){
                int tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
        }
    }
    
    private Instances[] splitByClass(Instances data){
        Instances[] dataSplitByClass;
        if (data.classAttribute().isNumeric()) {
            dataSplitByClass = new Instances[data.numClasses()];
            dataSplitByClass[0] = data;
        }else{
            dataSplitByClass = new Instances[data.numClasses() + 1];
            for (int i = 0; i < dataSplitByClass.length; i++) {
                dataSplitByClass[i] = new Instances(data, 0);
            }
            for (int i = 0; i < data.size(); i++) {
                if (data.get(i).classIsMissing()) {
                    dataSplitByClass[data.numClasses()].add(data.get(i));
                }else{
                    dataSplitByClass[(int)data.get(i).classValue()].add(data.get(i));
                }
            }
            //If there are no instances without a class then we can reduce array.
            if (dataSplitByClass[data.numClasses()].numInstances() == 0) {
                Instances[] temp = dataSplitByClass;
                dataSplitByClass = new Instances[data.numClasses()];
                System.arraycopy(temp, 0, dataSplitByClass, 0, data.numClasses());
            }
        }
        return dataSplitByClass;
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
    * Adds random instances to the dataset.
    * 
    * @param dataset the dataset
    * @param numInstances the number of instances
    * @param random a random number generator
    */
    protected void addRandomInstances( Instances dataset, int numInstances, Random random){
        int n = dataset.numAttributes();				
        double [] v = new double[ n ];
        for( int i = 0; i < numInstances; i++ ) {
            for( int j = 0; j < n; j++ ) {
                Attribute att = dataset.attribute( j );
                if( att.isNumeric() ) {
                    v[ j ] = random.nextDouble();
                }else {
                    if ( att.isNominal() ) { 
                        v[ j ] = random.nextInt( att.numValues() );
                    }   
                }  
            }
            dataset.add( new DenseInstance( 1, v ) );
        }
    }
    
    /** 
    * Transforms an instance for the i-th classifier.
    *
    * @param instance the instance to be transformed
    * @return the transformed instance
    * @throws Exception if the instance can't be converted successfully 
    */
    protected Instance convertInstance( Instance instance) throws Exception {
        Instance newInstance = new DenseInstance( attributeDetails.numAttributes());
        newInstance.setWeight(instance.weight());
        newInstance.setDataset(attributeDetails);
        int currentAttribute = 0;
        
        //Project the data using each group.
        for (int i = 0; i < groups.length; i++) {
            Instance temp = new DenseInstance(groups[i].length + 1);
            int j;
            for (j = 0; j < groups[i].length; j++) {
                temp.setValue(j, instance.value(groups[i][j]));
            }
            temp.setValue( j, instance.classValue( ) );
            temp.setDataset(reducedHeaders[i]);
            projectionFilters[i].input( temp );
            temp = projectionFilters[i].output( );
            projectionFilters[i].batchFinished();
            for( int a = 0; a < temp.numAttributes() - 1; a++ ) {
              newInstance.setValue( currentAttribute++, temp.value( a ) );
            }
        }
        newInstance.setClassValue(instance.classValue());
        return newInstance;
    }
}
