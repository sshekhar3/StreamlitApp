import graphviz

# Create a graphlib graph object

def drawGraph():
    graph1 = graphviz.Digraph()
    graph1.edge('Image', 'ML Model (YOLO model)')
    graph1.edge('Labels + Bounding box oordinates', 'ML Model (YOLO model)')
    graph1.edge('ML Model (YOLO model)', 'Training & Validation')
    graph1.edge('Training & Validation', 'Alphabet Prediction')
    graph1.edge('Alphabet Prediction', 'Text prediction')
    
    
    graph2 = graphviz.Digraph()
    graph2.edge('Image', 'Image processing & hand landmark extraction')
    graph2.edge('Image processing & hand landmark extraction', 'ML Model DNN Classifier')
    graph2.edge('Labels', 'ML Model DNN Classifier')
    graph2.edge('ML Model DNN Classifier', 'Training & Validation')
    graph2.edge('Training & Validation', 'Alphabet Prediction')
    graph2.edge('Alphabet Prediction', 'Text prediction')

    
    graph3 = graphviz.Digraph()
    graph3.edge('Current Video frame', 'Object detection')
    graph3.edge('Current Video frame', 'Hand landmarks extraction')
    graph3.edge('Hand landmarks extraction', 'Alphabet Prediction')
    graph3.edge('Alphabet Prediction', 'Next frame')
    graph3.edge('Object detection', 'Alphabet prediction')
    graph3.edge('Alphabet prediction', 'Next frame')
    graph3.edge('Next frame', 'Current Video frame')
    
    return graph1, graph2, graph3