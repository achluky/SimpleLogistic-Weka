import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;

import com.sun.xml.internal.bind.v2.schemagen.xmlschema.List;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class ClassifierSLR {
	/**
	 * Object that stores the instance.
	 */
	Instances instances;
	/**
	 * Object that stores the classifier.
	 */
	NaiveBayes classifier;
			
	/**
	 * This method loads the model to be used as classifier.
	 * @param fileName The name of the file that stores the model.
	 */
	public void loadModel(String fileName) {
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));
            Object tmp = in.readObject();
			classifier = (NaiveBayes) tmp;
            in.close();
 			System.out.println("===== Loaded model: " + fileName + " =====");
       } 
		catch (Exception e) {
			// Given the cast, a ClassNotFoundException must be caught along with the IOException
			System.out.println("Problem found when reading: " + fileName);
		}
	}
	
	/**
	 * This method creates the instance to be classified, from the text that has been read.
	 * @param csvInstance A CSV formatted instance of the Weather Numeric Dataset: "sunny,85,85,FALSE" or "sunny,85,85,FALSE,?"
	 */
	public void makeInstance(String csvInstance) {
		// Attributes are:
		// @attribute outlook {sunny, overcast, rainy}
		// @attribute temperature real
		// @attribute humidity real
		// @attribute windy {TRUE, FALSE}
		// @attribute play {yes, no}
		
		// Create the header
		List attributeList = (List) new ArrayList(5);
		
		// Atribute "outlook"
		List values = (List) new ArrayList(3); 
		values.add("sunny"); 
		values.add("overcast"); 
		values.add("rainy"); 
		Attribute attribute = new Attribute("outlook", values);
		attributeList.add(attribute);
		
		// Atribute "temperature" - default numeric
		attribute = new Attribute("temperature");
		attributeList.add(attribute);
		
		// Atribute "humidity"
		attribute = new Attribute("humidity");
		attributeList.add(attribute);
		
		// Atribute "windy"
		values = new ArrayList(2); 
		values.add("TRUE"); 
		values.add("FALSE"); 
		attribute = new Attribute("windy", values);
		attributeList.add(attribute);
		
		// Atribute "play"
		values = new ArrayList(2); 
		values.add("yes"); 
		values.add("no"); 
		attribute = new Attribute("play", values);
		attributeList.add(attribute);

		// Build instance set with just one instance
		instances = new Instances("Test relation", (java.util.ArrayList<Attribute>) attributeList, 1);           
		// Set class index
		instances.setClassIndex(instances.numAttributes()-1);
		
		
		// Create and add the instance
		DenseInstance instance = new DenseInstance(5);
		instance.setDataset(instances);
		
		// Assumed the instance is in CSV: "sunny,85,85,FALSE", class (last) undefined
		String[] stringValues = csvInstance.split(",");
		instance.setValue(0, stringValues[0]);
		instance.setValue(1, Integer.parseInt(stringValues[1]));
		instance.setValue(2, Integer.parseInt(stringValues[2]));
		instance.setValue(3, stringValues[3]);		
		instances.add(instance);
		
 		System.out.println("===== Instance created with reference dataset =====");
		System.out.println(instances);
	}
	
	/**
	 * This method performs the classification of the instance.
	 * Output is done at the command-line.
	 */
	public void classify() {
		try {
			double pred = classifier.classifyInstance(instances.instance(0));
			System.out.println("===== Classified instance =====");
			System.out.println("Class predicted: " + instances.classAttribute().value((int) pred));
		}
		catch (Exception e) {
			System.out.println("Problem found when classifying the example");
		}		
	}
	
	/**
	 * This method loads a dataset in ARFF format. If the file does not exist, or
	 * it has a wrong format, the attribute instances is null.
	 * @param fileName The name of the file that stores the dataset.
	 */
	public void loadHeader(String fileName) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			ArffReader arff = new ArffReader(reader);
			instances = arff.getData();
			System.out.println("===== Loaded dataset: " + fileName + " =====");
			reader.close();
		}
		catch (IOException e) {
			System.out.println("Problem found when reading: " + fileName);
		}
	}
	
	/**
	 * This method creates the instance to be classified, from the text that has been read.
	 * @param csvInstance A CSV formatted instance of the Weather Numeric Dataset: "sunny,85,85,FALSE" or "sunny,85,85,FALSE,?"
	 */
	public void makeInstanceFromHeader(String csvInstance) {
		// Attributes are:
		// @attribute outlook {sunny, overcast, rainy}
		// @attribute temperature real
		// @attribute humidity real
		// @attribute windy {TRUE, FALSE}
		// @attribute play {yes, no}

		// Set class index
		instances.setClassIndex(instances.numAttributes()-1);
		
		// Create and add the instance
		DenseInstance instance = new DenseInstance(5);
		instance.setDataset(instances);
		
		String[] stringValues = csvInstance.split(",");
		instance.setValue(0, stringValues[0]);
		instance.setValue(1, Integer.parseInt(stringValues[1]));
		instance.setValue(2, Integer.parseInt(stringValues[2]));
		instance.setValue(3, stringValues[3]);		
		instances.add(instance);
		
 		System.out.println("===== Instance created with reference dataset =====");
		System.out.println(instances);
	}	

	
	/**
	 * Main method. It is an example of the usage of this class.
	 * @param args Command-line arguments: csv-instance and fileModel.
	 */
	public static void main (String[] args) {
	
		ClassifierSLR classifier;
		if (args.length < 3) {
			System.out.println("Usage: java MyClassifier <csv-instance> <fileModel>");
			System.out.println("Or: java MyClassifier <csv-instance> <fileModel> <fileHeader>");
			System.out.println("Example: java MyClassifier \"sunny,85,85,FALSE,no\" myNaiveBayesModel.data");
			System.out.println("Or: java MyClassifier \"sunny,85,85,FALSE,no\" myNaiveBayesModel.data myWeatherHeader.arff");
		}
		else {
			classifier = new ClassifierSLR();
			classifier.loadModel(args[1]);
			classifier.makeInstance(args[0]);
			// classifier.loadHeader(args[2]);
			// classifier.makeInstanceFromHeader(args[0]);
			classifier.classify();
		}
	}
}
