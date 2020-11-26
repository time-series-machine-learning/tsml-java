package tsml.transformers;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import experiments.data.DatasetLoading;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.ts_fileIO.TSReader;
import utilities.InstanceTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class CombineTransformer implements Transformer {

    Transformer[] transforms;

    public CombineTransformer(List<Transformer> transformers) {
        this((Transformer[]) transformers.toArray());
    }

    public CombineTransformer(Transformer[] transformers) {
        transforms = transformers; // TODO: could deep copy here
    }

    @Override
    public Instance transform(Instance inst) {
        List<Double> data = new ArrayList<>();
        for (Transformer trans : transforms) {
            Instance out = trans.transform(inst);
            // TODO: Change: we assume the class value is at the end.
            for (double d : InstanceTools.ConvertInstanceToArrayRemovingClassValue(out, out.numAttributes() - 1))
                data.add(d);
        }

        // put the class value on the end.
        data.add(inst.classValue());

        // this is the cleanest way to convert from a List<Double> to double[]. can't
        // cast etc, have to unpack.
        Instance out = new DenseInstance(1.0, data.stream().mapToDouble(Double::doubleValue).toArray());
        return out;
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {

        //initiliase empty container.
        List<List<Double>> data = new ArrayList<>();
        for(int i=0; i< inst.getNumDimensions(); i++){
            data.add(new ArrayList<>());
        }

        for (Transformer trans : transforms) {
            TimeSeriesInstance ts = trans.transform(inst);

            //append the data to the end of the 2D array. Yuck!
            double[][] out = ts.toValueArray();
            for(int i=0; i< out.length; ++i)
                for (double d : out[i])
                    data.get(i).add(d);
        }

        return new TimeSeriesInstance(data, inst.getLabelIndex(), inst.getClassLabels());
    }

    @Override
    public Instances determineOutputFormat(Instances inputFormat) throws IllegalArgumentException {
        ArrayList<Attribute> atts = new ArrayList<>();
        String transform_names = "Concat";
        for (Transformer trans : transforms) {

            Instances out = trans.determineOutputFormat(inputFormat);
            transform_names += " | " + out.relationName();
            for (int i = 0; i < out.numAttributes(); i++) {
                if (out.classIndex() == i)
                    continue; // skip class index.

                atts.add(new Attribute("Concat_" + out.attribute(i).name()));
            }
        }

        System.out.println(transform_names);

        if (inputFormat.classIndex() >= 0) { // Classification set, set class
            // Get the class values as a fast vector
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            ArrayList<String> vals = new ArrayList<>();
            for (int i = 0; i < target.numValues(); i++)
                vals.add(target.value(i));
            atts.add(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }
        Instances result = new Instances(transform_names, atts, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0) {
            result.setClassIndex(result.numAttributes() - 1);
        }
        return result;
    }

    public static void main(String[] args) throws FileNotFoundException, IOException {

        String local_path = "D:\\Work\\Data\\Univariate_ts\\"; // Aarons local path for testing.
        String dataset_name = "ChinaTown";

        File f1 = new File(local_path + dataset_name + File.separator + dataset_name + "_TRAIN.ts");
        TSReader ts_reader_multi = new TSReader(new FileReader(f1));
        TimeSeriesInstances train = ts_reader_multi.GetInstances();

        CombineTransformer combined = new CombineTransformer(new Transformer[] { new Cosine(), new Sine() });
        System.out.println(train);
        System.out.println(combined.transform(train));

    }

}
