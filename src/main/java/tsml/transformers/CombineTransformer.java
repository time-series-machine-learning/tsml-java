package tsml.transformers;

import java.io.File;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import experiments.data.DatasetLoading;
import utilities.InstanceTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class CombineTransformer implements Transformer {


    Transformer[] transforms;

    public CombineTransformer(List<Transformer> transformers){
        this((Transformer[]) transformers.toArray());
    }

    public CombineTransformer(Transformer[] transformers) {
        transforms = transformers; //TODO: could deep copy here
    }


    @Override
    public Instance transform(Instance inst) {
        List<Double> data = new ArrayList<>();
        for(Transformer trans : transforms){
            Instance out = trans.transform(inst);
            //TODO: Change: we assume the class value is at the end. 
            for(double d : InstanceTools.ConvertInstanceToArrayRemovingClassValue(out, out.numAttributes()-1))
                data.add(d);
        }

        //put the class value on the end.
        data.add(inst.classValue());

        //this is the cleanest way to convert from a List<Double> to double[]. can't cast etc, have to unpack.
        Instance out = new DenseInstance(1.0, data.stream().mapToDouble(Double::doubleValue).toArray());
        return out;
    }

    @Override
    public Instances determineOutputFormat(Instances inputFormat) throws IllegalArgumentException {
        ArrayList<Attribute> atts = new ArrayList<>();
        String transform_names = "Concat";
        for(Transformer trans : transforms){
            
            Instances out = trans.determineOutputFormat(inputFormat);
            transform_names += " | " + out.relationName();
            for(int i=0; i<out.numAttributes(); i++){
                if(out.classIndex() == i) continue; //skip class index.
                
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
    

    public static void main(String[] args){

        String local_path = "D:\\Work\\Data\\Univariate_ts\\"; //Aarons local path for testing.
        String dataset_name = "ChinaTown";
        Instances train = DatasetLoading.loadData(local_path + dataset_name + File.separator + dataset_name+"_TRAIN.ts");
        Instances test  = DatasetLoading.loadData(local_path + dataset_name + File.separator + dataset_name+"_TEST.ts");
        CombineTransformer combined = new CombineTransformer(new Transformer[]{new Cosine(), new Sine()});
        System.out.println(train);
        System.out.println(combined.transform(train));



    }
}