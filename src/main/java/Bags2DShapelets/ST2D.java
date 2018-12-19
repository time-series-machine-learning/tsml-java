package Bags2DShapelets;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class ST2D extends SimpleBatchFilter {

    ShapeletSearch2D searcher;
    Shapelet2D[] shapelets;
            
    int k = -1;
    int numShapeletsToSearch = 1000;
    
    boolean shapeletsFound = false;
    
    public ST2D() {
        searcher = new ShapeletSearch2D(0);
    }
    
    public ST2D(int seed) {
        searcher = new ShapeletSearch2D(seed);
    }
    
    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
        //copied from shapelettransform
        
        if (k < 1) {
            System.out.println(this.k);
            throw new IllegalArgumentException("ShapeletFilter not initialised correctly - please specify a value of k that is greater than or equal to 1");
        }

 
        FastVector atts = new FastVector();
        String name;
        for (int i = 0; i < k; i++) {
            name = "Shapelet_" + i;
            atts.addElement(new Attribute(name));
        }

        if (inputFormat.classIndex() >= 0) {
            //Classification set, set class
            //Get the class values as a fast vector
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            FastVector vals = new FastVector(target.numValues());
            for (int i = 0; i < target.numValues(); i++) {
                vals.addElement(target.value(i));
            }
            atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }
        Instances result = new Instances("Shapelets" + inputFormat.relationName(), atts, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0) {
            result.setClassIndex(result.numAttributes() - 1);
        }
        return result;
    }

    @Override
    public Instances process(Instances imgs) throws Exception {
        
        //checks if the shapelets haven't been found yet, finds them if it needs too.
        if (!shapeletsFound) {
            searcher.numShapeletsToSearch = numShapeletsToSearch;
            if (k > 0) // has already been set
                searcher.k = this.k;
            
            shapelets = searcher.generateKShapelets(imgs);
            if (k < 1) // if wasnt already set before, searcher will have set k to numinstances
                k = shapelets.length;
            shapeletsFound = true;
        }

        //build the transformed dataset with the shapelets we've found either on this data, or the previous training data
        return transformDataset(imgs);
        
    }

    public Instances transformDataset(Instances imgs) throws Exception {
        Instances transformed = determineOutputFormat(imgs);
        
        int numInsts = imgs.numInstances();
        int numAtts = shapelets.length + 1; // + classVal
        
        //create the (empty) instances
        for (int instId = 0; instId < numInsts; instId++) {
            transformed.add(new DenseInstance(shapelets.length + 1));
        }

        //populate the distances
        for (int instId = 0; instId < numInsts; instId++) {
            for (int shapeletID = 0; shapeletID < shapelets.length; shapeletID++) {
                double distance = searcher.sDist2D(imgs.get(instId), shapelets[shapeletID]);
                transformed.instance(instId).setValue(shapeletID, distance);
            }
        }
        
        //do the classValues
        for (int instId = 0; instId < numInsts; instId++)
            transformed.instance(instId).setValue(numAtts-1, imgs.instance(instId).classValue());
        
        return transformed;
    }
    
}
