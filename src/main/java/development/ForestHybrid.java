package development;
        
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import utilities.ClassifierResults;
import utilities.SaveParameterInfo;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.instance.RemovePercentage;

public class ForestHybrid extends AbstractClassifier implements SaveParameterInfo{

    private Classifier[] classifiers = null;
    private int[] startIndex = null;
    private int[][] attIndexs = null;
    private String classifier = "";
    private int featureSpace = 0;
    private int transformation = 0;
    private int numTrees = 200;
    private int seed = 0;
    private Random random;
    private String relationName;
    private ClassifierResults res =new ClassifierResults();
    
    //Specific to RotationForest
    private int minGroup = 3;
    private int maxGroup = 3;
    private RemoveUseless removeUseless = null;
    private Normalize normalize = null;
    private int groups[][][];
    private Instances [] headers = null;
    private Instances [][] reducedHeaders = null;
    private Filter [][] projectionFilters = null;
    private Filter projectionFilter = null;
    private int removedPercentage = 50;
    private Instances trainingData;
    
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        long startTime=System.currentTimeMillis();
        Instances [] instancesOfClass = null;
        classifiers = new Classifier[numTrees];
        startIndex = new int[numTrees];
        attIndexs = new int[numTrees][(int)Math.sqrt(instances.numAttributes()-1)];
        relationName = instances.relationName();
        
        
        if (transformation != 0) {
            projectionFilter = defaultFilter();

            checkMinMax(instances);

            if( instances.numInstances() > 0 ) {
                // This function fails if there are 0 instances
                random = instances.getRandomNumberGenerator(seed);
            }

            removeUseless = new RemoveUseless();
            removeUseless.setInputFormat(instances);
            instances = Filter.useFilter(instances, removeUseless);

            normalize = new Normalize();
            normalize.setInputFormat(instances);
            instances = Filter.useFilter(instances, normalize);

            generateGroupsFromNumbers(instances, random);

            projectionFilters = new Filter[groups.length][];
            for(int i = 0; i < projectionFilters.length; i++ ) {
                projectionFilters[i] = Filter.makeCopies( projectionFilter, groups[i].length );
            }

            int numClasses = instances.numClasses();

            // Split the instances according to their class
            instancesOfClass = new Instances[numClasses + 1]; 
            if( instances.classAttribute().isNumeric() ) {
                instancesOfClass = new Instances[numClasses]; 
                instancesOfClass[0] = instances;
            }
            else {
                instancesOfClass = new Instances[numClasses+1]; 
                for( int i = 0; i < instancesOfClass.length; i++ ) {
                    instancesOfClass[ i ] = new Instances( instances, 0 );
                }
                Enumeration enu = instances.enumerateInstances();
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

            // These arrays keep the information of the transformed data set
            headers = new Instances[ classifiers.length ];
            reducedHeaders = new Instances[ classifiers.length ][];
        }
        
        for (int i = 0; i < numTrees; i++) {
            trainingData = instances;
            if (transformation != 0) {
                reducedHeaders[i] = new Instances[ groups[i].length ];
                FastVector transformedAttributes = new FastVector( instances.numAttributes() );

                // Construction of the dataset for each group of attributes
                for( int j = 0; j < groups[ i ].length; j++ ) {
                    FastVector fv = new FastVector( groups[i][j].length + 1 );
                    for( int k = 0; k < groups[i][j].length; k++ ) {
                        String newName = instances.attribute( groups[i][j][k] ).name() + "_" + k;
                        //System.out.println(newName);
                        fv.addElement( instances.attribute( groups[i][j][k] ).copy(newName) );
                    }
                    fv.addElement( instances.classAttribute( ).copy() );
                    Instances dataSubSet = new Instances( "rotated-" + i + "-" + j + "-", fv, 0);
                    
                    
                    dataSubSet.setClassIndex( dataSubSet.numAttributes() - 1 );

                    // Select instances for the dataset
                    reducedHeaders[i][j] = new Instances( dataSubSet, 0 );
                    boolean [] selectedClasses = selectClasses( instancesOfClass.length, random );
                    for( int c = 0; c < selectedClasses.length; c++ ) {
                        if( !selectedClasses[c] )
                            continue;
                        Enumeration enu = instancesOfClass[c].enumerateInstances();
                        while( enu.hasMoreElements() ) {
                            Instance instance = (Instance)enu.nextElement();
                            Instance newInstance = new DenseInstance(dataSubSet.numAttributes());
                            newInstance.setDataset( dataSubSet );
                            for( int k = 0; k < groups[i][j].length; k++ ) {
                                newInstance.setValue( k, instance.value( groups[i][j][k] ) );
                            }
                            newInstance.setClassValue( instance.classValue( ) );
                            dataSubSet.add( newInstance );
                        }
                    }

                    if(transformation == 2){
                        //System.out.println("Remove percentage");
                        dataSubSet.randomize(random);
                        // Remove a percentage of the instances
                        Instances originalDataSubSet = dataSubSet;
                        dataSubSet.randomize(random);
                        RemovePercentage rp = new RemovePercentage();
                        rp.setPercentage( removedPercentage );
                        rp.setInputFormat( dataSubSet );
                        dataSubSet = Filter.useFilter( dataSubSet, rp );
                        if( dataSubSet.numInstances() < 2 ) {
                            dataSubSet = originalDataSubSet;
                        }
                    }


                    // Project de data
                    projectionFilters[i][j].setInputFormat( dataSubSet );
                    Instances projectedData = null;
                    do {
                        try {
                            projectedData = Filter.useFilter( dataSubSet, projectionFilters[i][j] );
                        } catch ( Exception e ) {
                            // The data could not be projected, we add some random instances
                            addRandomInstances( dataSubSet, 10, random );
                        }
                    } while( projectedData == null );

                    // Include the projected attributes in the attributes of the 
                    // transformed dataset
                    for( int a = 0; a < projectedData.numAttributes() - 1; a++ ) {
                        String newName = projectedData.attribute(a).name() + "_" + j;
                        //System.out.println(newName);
                        transformedAttributes.addElement( projectedData.attribute(a).copy(newName));
                    }
                }

                transformedAttributes.addElement( instances.classAttribute().copy() );
                Instances buildClas = new Instances( "rotated-" + i + "-", transformedAttributes, 0 );
                buildClas.setClassIndex( buildClas.numAttributes() - 1 );
                headers[ i ] = new Instances( buildClas, 0 );

                  // Project all the training data
                Enumeration enu = instances.enumerateInstances();
                while( enu.hasMoreElements() ) {
                    Instance instance = (Instance)enu.nextElement();
                    Instance newInstance = convertInstance( instance, i );
                    buildClas.add( newInstance );
                }

                  // Build the base classifier
//                if (classifiers[0] instanceof Randomizable) {
//                    ((Randomizable) classifiers[i]).setSeed(random.nextInt());
//                }
                
                trainingData = buildClas;
            }


            classifiers[i] = getClassifier();

            if(featureSpace == 0){

                if(classifier.equalsIgnoreCase("RandomTree")){
                    ((RandomTree)classifiers[i]).setKValue((int)Math.sqrt(trainingData.numAttributes()-1));
                } else if(classifier.equalsIgnoreCase("J48")){
                    trainingData = featureReduction(trainingData, i);
                }   
                
            }else if(featureSpace == 1){
                
                if(classifier.equalsIgnoreCase("RandomTree")){
                    ((RandomTree)classifiers[i]).setKValue(trainingData.numAttributes()-1);
                } else if(classifier.equalsIgnoreCase("J48")){

                }
                
            }
            
//            for (int j = 0; j < 4; j++) {
//                System.out.println(trainingData.get(j));
//            }
            
            
            classifiers[i].buildClassifier(trainingData);
        }
        res.buildTime=System.currentTimeMillis()-startTime;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[]distribution = distributionForInstance(instance);

        return getMaxVoteFromDist(distribution);
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception{
        if(transformation == 0){
            double [] sums = new double[instance.numClasses()], newProbs;
            Instance temp = null;
            for (int i = 0; i < this.numTrees; i++) {
                if(featureSpace == 0 && classifier.equalsIgnoreCase("J48")){
                    temp = produceIntervalInstance(instance, i);
                    newProbs = classifiers[i].distributionForInstance(temp);
                    for (int j = 0; j < newProbs.length; j++)
                        sums[j] += newProbs[j];
                }else{
                    newProbs = classifiers[i].distributionForInstance(instance);
                    for (int j = 0; j < newProbs.length; j++)
                        sums[j] += newProbs[j];
                }  
            }
            Utils.normalize(sums);
            return sums;
        }else{
            
            removeUseless.input(instance);
            instance =removeUseless.output();
            removeUseless.batchFinished();

            normalize.input(instance);
            instance = normalize.output();
            normalize.batchFinished();

            double [] sums = new double [instance.numClasses()], newProbs; 

            for (int i = 0; i < classifiers.length; i++) {
                
                Instance convertedInstance = convertInstance(instance, i);
                
                if(featureSpace == 0){

                    if(classifier.equalsIgnoreCase("J48")){
                        convertedInstance = produceIntervalInstance(convertedInstance, i);
                    }   

                }
                
                if (convertedInstance.classAttribute().isNumeric() == true) {
                    sums[0] += classifiers[i].classifyInstance(convertedInstance);
                } else {
                    newProbs = classifiers[i].distributionForInstance(convertedInstance);
                    for (int j = 0; j < newProbs.length; j++)
                    sums[j] += newProbs[j];
                }
            }
            if (instance.classAttribute().isNumeric() == true) {
                sums[0] /= (double)numTrees;
                return sums;
            } else if (Utils.eq(Utils.sum(sums), 0)) {
                return sums;
            } else {
                Utils.normalize(sums);
                return sums;
            }
        }
    }
    
    private int getMaxVoteFromDist(double[] distribution){
        int maxVote=0;
        for(int i = 1; i < distribution.length; i++)
            if(distribution[i] > distribution[maxVote])
                maxVote = i;
        return maxVote;
    }
    
    private Classifier getClassifier(){
        Classifier c = null;
        if(classifier.equalsIgnoreCase("RandomTree")){
            c = new RandomTree();
            ((RandomTree)c).setSeed(random.nextInt());
        }
        if(classifier.equalsIgnoreCase("J48")){
            c = new J48();
            ((J48)c).setSeed(random.nextInt());
        }
        return c;
    }
    
    public Instances featureReduction(Instances trainInstances, int i){
        Instances temp = new Instances(trainInstances);
        Random rand = new Random(seed);
        List<Integer> indx = new ArrayList<>();
        for (int j = 0; j < temp.numAttributes()-1; j++) {
            indx.add(j);
        }
        Collections.shuffle(indx, rand);
        for (int j = 0; j < attIndexs[i].length; j++) {
            //attIndexs[i][j] = indx.get(j);
            int x = rand.nextInt(indx.size()-0);
            try{
                attIndexs[i][j] = indx.get(x);
            }catch(Exception e){
                attIndexs[i][j] = indx.get(x);
            }
        }
        
        //Instances temp;
        //startIndex[i] = rand.nextInt((trainInstances.numAttributes()) - (int)Math.sqrt(trainInstances.firstInstance().numAttributes() - 1));
        
        temp = produceIntervalInstances(temp, i);
        return temp;
    }
    
    private Instances produceIntervalInstances(Instances trainInstances, int i){
        
        //POPULATE INTERVAL INSTANCES. 
        //Create and populate attribute information based on interval, class attribute is an addition.
        ArrayList<Attribute>attributes = new ArrayList<>();
        Random r = new Random(seed);
        for (int j = 0; j < attIndexs[i].length ; j++) {
            attributes.add(trainInstances.attribute(attIndexs[i][j]).copy(trainInstances.attribute(attIndexs[i][j]).name()+r.nextInt()));
        }
        attributes.add(trainInstances.attribute(trainInstances.numAttributes()-1));

        //Create new Instances to hold intervals.
        String relationName = trainInstances.relationName();
        Instances intervalInstances;
        intervalInstances = new Instances(relationName, attributes, trainInstances.size());
        double[] temp = null;
            for (int k = 0; k < trainInstances.size(); k++) {
            //Produce intervals from input instances, additional attribute needed to accomidate class value.
            //double[] temp = Arrays.copyOfRange(trainInstances.get(k).toDoubleArray(), startIndex[i], startIndex[i] + (int)Math.sqrt(trainInstances.firstInstance().numAttributes()-1) + 1);
            temp = new double[attIndexs[i].length+1];
            for (int j = 0; j < temp.length-1; j++) {
                temp[j] = trainInstances.get(k).value(attIndexs[i][j]);
            }
            DenseInstance instance = new DenseInstance(temp.length);
            instance.replaceMissingValues(temp);
            instance.setValue(temp.length-1, trainInstances.get(k).classValue());
            intervalInstances.add(instance);
        }
        intervalInstances.setClassIndex(temp.length-1);

        return intervalInstances;
    }
    
    private Instance produceIntervalInstance(Instance instance, int classifierNum){
        
        ArrayList<Attribute>attributes = new ArrayList<>();
        for (int i = 0; i < instance.numAttributes(); i++) {
            attributes.add(instance.attribute(i));
        }
        Instances intervalInstances = new Instances(relationName, attributes, 1);
        intervalInstances.add(instance);
        intervalInstances.setClassIndex(instance.numAttributes()-1);
        intervalInstances = produceIntervalInstances(intervalInstances, classifierNum);
        
        return intervalInstances.firstInstance();
    }
    
    public void setClassifier(String treeName){
        switch(treeName.toLowerCase()){
            case "randomtree": 
                classifier = "RandomTree";
                break;
            case "j48": 
                classifier = "J48";
                break;
            default:
                classifier = "RandomTree";
                break;
        }
    }
    
    public void setFeatureSpace(int i){
        switch(i){
            case 0:
                featureSpace = 0;
                break;
            case 1:
                featureSpace = 1;
                break;
            default:
                featureSpace = 0;
                break;
        }
    }
    
    public void setTransformType(int i){
        switch(i){
            case 0:
                transformation = 0;
                break;
            case 1:
                transformation = 1;
                break;
            case 2:
                transformation = 2;
                break;
            default:
                transformation = 0;
                break;
        }
    }
    
    public void setSeed(int i){
        seed = i;
        random = new Random(seed);
    }
    
    public void setNumTrees(int i){
        numTrees = i;
    }
    
    //ROTATION FOREST SPECIFIC METHODS
    
    protected Filter defaultFilter() {
        PrincipalComponents filter = new PrincipalComponents();
        //filter.setNormalize(false);
        filter.setVarianceCovered(1.0);
        return filter;
    }
    
    private void checkMinMax(Instances data) {
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
    
    protected void generateGroupsFromNumbers(Instances data, Random random) {
        groups = new int[classifiers.length][][];
        for( int i = 0; i < classifiers.length; i++ ) {
            int [] permutation = attributesPermutation(data.numAttributes(), data.classIndex(), random);
            int numGroups = minGroup + random.nextInt(maxGroup - minGroup + 1);
            groups[i] = new int[numGroups][];
            int groupSize = permutation.length / numGroups;

            // Some groups will have an additional attribute
            int numBiggerGroups = permutation.length % numGroups;

            // Distribute the attributes in the groups
            int currentAttribute = 0;
            for( int j = 0; j < numGroups; j++ ) {
                if( j < numBiggerGroups ) {
                    groups[i][j] = new int[groupSize + 1];
                }
                else {
                    groups[i][j] = new int[groupSize];
                }
                for( int k = 0; k < groups[i][j].length; k++ ) {
                    groups[i][j][k] = permutation[currentAttribute++];
                }
            }
        }
    }
    
    protected int [] attributesPermutation(int numAttributes, int classAttribute,Random random) {
        int [] permutation = new int[numAttributes-1];
        int i = 0;
        for(; i < classAttribute; i++){
            permutation[i] = i;
        }
        for(; i < permutation.length; i++){
            permutation[i] = i + 1;
        }

        permute( permutation, random );

        return permutation;
    }
    
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
        
        if(transformation == 1){
            //System.out.println("No class removal");
            for (int i = 0; i < selected.length; i++) {
                selected[i] = true;
            }
        }
        
        return selected;
    }
    
    protected void addRandomInstances( Instances dataset, int numInstances, Random random ) {
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
    
    protected Instance convertInstance( Instance instance, int i ) throws Exception {
        Instance newInstance = new DenseInstance( headers[ i ].numAttributes( ) );
        newInstance.setWeight(instance.weight());
        newInstance.setDataset( headers[ i ] );
        int currentAttribute = 0;

        // Project the data for each group
        for( int j = 0; j < groups[i].length; j++ ) {
            Instance auxInstance = new DenseInstance( groups[i][j].length + 1 );
            int k;
            for( k = 0; k < groups[i][j].length; k++ ) {
                auxInstance.setValue( k, instance.value( groups[i][j][k] ) );
            }
            auxInstance.setValue( k, instance.classValue( ) );
            auxInstance.setDataset( reducedHeaders[ i ][ j ] );
            projectionFilters[i][j].input( auxInstance );
            auxInstance = projectionFilters[i][j].output( );
            projectionFilters[i][j].batchFinished();
            for( int a = 0; a < auxInstance.numAttributes() - 1; a++ ) {
                newInstance.setValue( currentAttribute++, auxInstance.value( a ) );
            }
        }
        newInstance.setClassValue( instance.classValue() );
        return newInstance;
    }
    
    //--------------------------------------------------------------------------

    @Override
    public String getParameters() {
        String result = "BuildTime," + res.buildTime + ",CVAcc," + res.acc + ",";
        result += "MaxDepth," + getMaxDepth() + ",NumFeatures," + trainingData.numAttributes() + ",NumTrees," + numTrees;
        return result;
    }
    
    private double getMaxDepth(){
        Classifier c;
        if (classifier.equalsIgnoreCase("J48")) {
            return 0;
        }else{
            return ((RandomTree)classifiers[0]).getMaxDepth();
        }
    }
    
    public static void main(String[] args) {
        
    }
}