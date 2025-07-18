package org.processmining.specpp.prom.mvc.result;

import com.fluxicon.slickerbox.factory.SlickerFactory;
import org.deckfour.xes.model.XLog;
import org.processmining.framework.plugin.PluginContext;
import org.processmining.models.connections.petrinets.behavioral.FinalMarkingConnection;
import org.processmining.models.connections.petrinets.behavioral.InitialMarkingConnection;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.semantics.petrinet.Marking;
import org.processmining.plugins.utils.ProvidedObjectHelper;
import org.processmining.specpp.base.Evaluator;
import org.processmining.specpp.config.InputProcessingConfig;
import org.processmining.specpp.datastructures.encoding.BitMask;
import org.processmining.specpp.datastructures.log.Log;
import org.processmining.specpp.datastructures.log.Variant;
import org.processmining.specpp.datastructures.log.impls.IndexedVariant;
import org.processmining.specpp.datastructures.petri.CollectionOfPlaces;
import org.processmining.specpp.datastructures.petri.Place;
import org.processmining.specpp.datastructures.petri.ProMPetrinetWrapper;
import org.processmining.specpp.evaluation.fitness.results.DetailedFitnessEvaluation;
import org.processmining.specpp.prom.mvc.AbstractStagePanel;
import org.processmining.specpp.prom.mvc.config.ProMConfig;
import org.processmining.specpp.prom.plugins.ProMSPECppConfig;

import javax.swing.*;

public class ResultExportPanel extends AbstractStagePanel<ResultController> {

    private final PluginContext context;
    private final JButton saveProMPetriButton, saveConfigButton, saveEvalLogButton, placeInfoButton;
    private final CollectionOfPlaces colPlaces;

    public ResultExportPanel(ResultController controller, CollectionOfPlaces colPlaces) {
        super(controller);
        context = this.controller.getContext();
        setLayout(new BoxLayout(this, BoxLayout.X_AXIS));
        this.colPlaces = colPlaces;

        saveProMPetriButton = SlickerFactory.instance().createButton("save Petri net to workspace");
        saveProMPetriButton.addActionListener(e -> saveProMPetri());
        add(saveProMPetriButton);
        saveConfigButton = SlickerFactory.instance().createButton("save config to workspace");
        saveConfigButton.addActionListener(e -> saveConfig());
        add(saveConfigButton);

        saveEvalLogButton = SlickerFactory.instance().createButton("save evaluation log to workspace");
        saveEvalLogButton.addActionListener(e -> saveEvalLog());
        if (controller.createEvalLog() == controller.getRawLog()) saveEvalLogButton.setVisible(false);
        add(saveEvalLogButton);

        placeInfoButton = SlickerFactory.instance().createButton("save Place information");
        placeInfoButton.addActionListener(e -> savePlaceInfo());
        if (colPlaces==null) placeInfoButton.setEnabled(false);
        add(placeInfoButton);
    }

    private void saveProMPetri() {
        ProMPetrinetWrapper result = controller.getResult();
        Petrinet net = result.getNet();
        ProvidedObjectHelper.publish(context, "Petrinet", net, Petrinet.class, true);
        context.getConnectionManager().addConnection(new InitialMarkingConnection(net, result.getInitialMarking()));
        context.getConnectionManager()
               .addConnection(new FinalMarkingConnection(net, result.getFinalMarkings()
                                                                    .stream()
                                                                    .findFirst()
                                                                    .orElse(new Marking())));
        saveProMPetriButton.setEnabled(false);
    }

    private void saveConfig() {
        ProMConfig proMConfig = controller.getParentController().getProMConfig();
        InputProcessingConfig inputProcessingConfig = controller.getParentController().getInputProcessingConfig();
        ProMSPECppConfig config = new ProMSPECppConfig(inputProcessingConfig, proMConfig);
        ProvidedObjectHelper.publish(context, "SPECpp Config", config, ProMSPECppConfig.class, true);
        saveConfigButton.setEnabled(false);
    }

    private void saveEvalLog() {
        XLog evalLog = controller.createEvalLog();
        ProvidedObjectHelper.publish(context, "Evaluation Log", evalLog, XLog.class, true);
        saveEvalLogButton.setEnabled(false);
    }

    private void savePlaceInfo() {
        Evaluator<Place, DetailedFitnessEvaluation> fitEval = controller.getFitnessEvaluator();
        Object[] colPlacesArray = colPlaces.getPlaces().toArray();
        Log usedLog = controller.getLog();

        String placeInfo = "{\"place_info\":{";
        for (int i = 0; i < colPlaces.getPlaces().size(); i++) {
            if (i%5000==0) System.out.println("DEBUG: Saving place " + i + "/" + colPlaces.getPlaces().size());
            DetailedFitnessEvaluation value = fitEval.apply((Place) colPlacesArray[i]);
            BitMask fittingVariants = value.getFittingVariants();
            placeInfo += "\"" + colPlacesArray[i] + "\":\"" + fittingVariants + "\"";
            if (i != colPlaces.getPlaces().size()-1) placeInfo += ",";
        }
        System.out.println("DEBUG: Saving places done! Starting with variant Info!");
        placeInfo += "},\"variant_info\":{";
        boolean firstVar = true;
        for (IndexedVariant indexedVariant : usedLog) {
            // Add , before every new entry, except the first one
            if (!firstVar) {
                placeInfo += ",";
            }
            firstVar = false;
            Variant v = indexedVariant.getVariant();
            int varIndex = indexedVariant.getIndex();
            int varFreq = usedLog.getVariantFrequency(varIndex);
            placeInfo += "\"" + varIndex + "\":{\"var\":\"" + v.toString() + "\",\"freq\":" + varFreq + "}";
        }
        System.out.println("DEBUG: Variant Info done!");
        placeInfo += "}}";
        ProvidedObjectHelper.publish(context, "Place Clustering Info", placeInfo, String.class, true);
        placeInfoButton.setEnabled(false);
    }

}
