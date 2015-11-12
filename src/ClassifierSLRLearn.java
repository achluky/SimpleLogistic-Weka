import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SimpleLogistic;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
/**
 * 
 * @author ahmadluky
 *
 */
public class ClassifierSLRLearn {

	Instances trainData;
	SimpleLogistic classifier;
	public void loadDataset(String fileName) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			ArffReader arff = new ArffReader(reader);
			trainData = arff.getData();
			System.out.println("===== Loaded dataset: " + fileName + " =====");
			reader.close();
		}
		catch (IOException e) {
			System.out.println("Problem found when reading: " + fileName);
		}
	}
	
	public void evaluate() {
		try {
			trainData.setClassIndex(trainData.numAttributes()-1);
			classifier = new SimpleLogistic();
			Evaluation eval = new Evaluation(trainData);
			// best k=8 for cross validation
			eval.crossValidateModel(classifier, trainData, 8, new Random(1));
			System.out.println(eval.toSummaryString());
			System.out.println(eval.toClassDetailsString());
			System.out.println("===== Evaluating on filtered (training) dataset done =====");
		}
		catch (Exception e) {
			System.out.println("Problem found when evaluating");
		}
	}
	
	public void learn() {
		try {
			trainData.setClassIndex(trainData.numAttributes()-1);
			classifier = new SimpleLogistic();
			classifier.buildClassifier(trainData);
			System.out.println(classifier);
			System.out.println("===== Training on filtered (training) dataset done =====");
		}
		catch (Exception e) {
			System.out.println("Problem found when training");
		}
	}
	
	public void saveModel(String fileName) {
		try {
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName));
            out.writeObject(classifier);
            out.close();
 			System.out.println("===== Saved model: " + fileName + " =====");
        } 
		catch (IOException e) {
			System.out.println("Problem found when writing: " + fileName);
		}
	}
	
	public static void main (String[] args) {
		String fl = "./iris.arff";
		ClassifierSLRLearn learner;
		learner = new ClassifierSLRLearn();
		learner.loadDataset(fl);
		learner.evaluate();
		learner.learn();
		learner.saveModel("./model-iris.dat");
	}
}
