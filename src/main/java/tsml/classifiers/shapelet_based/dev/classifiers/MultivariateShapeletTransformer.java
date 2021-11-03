package tsml.classifiers.shapelet_based.dev.classifiers;

import tsml.classifiers.shapelet_based.dev.filter.ShapeletFilterMV;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import tsml.transformers.TrainableTransformer;
import weka.core.*;

import java.util.ArrayList;
import java.util.List;

public class MultivariateShapeletTransformer implements TrainableTransformer, Randomizable {

    private boolean normalise = true;
    private boolean fit = false;
    private int seed;

    private ShapeletFilterMV filter;
    private List<ShapeletMV> shapelets;
    private Instances transformData;


    private MSTC.ShapeletParams params;

    public MultivariateShapeletTransformer(MSTC.ShapeletParams params){
        this.params = params;
    }
    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        ArrayList<Attribute> atts = new ArrayList<>();
        for (int i = 0; i < shapelets.size(); i++) {
            atts.add(new Attribute("att" + i));
        }
        if (data.classIndex() >= 0) atts.add(data.classAttribute());
        Instances transformedData = new Instances("ShapeletTransform", atts, data.numInstances());
        if (data.classIndex() >= 0) transformedData.setClassIndex(transformedData.numAttributes() - 1);
        return transformedData;
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        ShapeletFunctions fun = params.type.createShapeletType();
        int size = shapelets.size();

        double[] data = new double[size];
        double dist;
        int i=0;
        for (ShapeletMV shapelet: this.shapelets) {
            data[i] = fun.sDist(shapelet,inst);
            i++;
        }
        ArrayList<TimeSeries> ts = new ArrayList<TimeSeries>();
        ts.add(new TimeSeries(data));
        return new TimeSeriesInstance(inst.getLabelIndex(), ts);
    }

    @Override
    public Instance transform(Instance inst) {
        ShapeletFunctions fun = params.type.createShapeletType();
        int size = shapelets.size();
        Instance out = new DenseInstance(size + 1);
        out.setDataset(transformData);
        double dist;
        int i=0;
        for (ShapeletMV shapelet: this.shapelets) {
            dist = fun.sDist(shapelet, Converter.fromArff(inst));
            out.setValue(i, dist);

            i++;
        }
        return out;
    }

    @Override
    public void fit(TimeSeriesInstances data) {
        this.filter = params.filter.createFilter();
        filter.setHourLimit(params.contractTimeHours);
        shapelets = filter.findShapelets(params, data);
        fit = true;

    }

    @Override
    public void fit(Instances data) {
        this.fit(Converter.fromArff(data));
    }

    @Override
    public boolean isFit() {
        return fit;
    }


    @Override
    public void setSeed(int seed) {
        this.seed = seed;
    }

    @Override
    public int getSeed() {
        return seed;
    }

    public String getParameters(){
        return filter.getParameters().toString();
    }
}
