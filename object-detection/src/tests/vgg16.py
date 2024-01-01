def test():
    from ..models import SSD_VGG16
    import torch
    from torchview import draw_graph

    num_bboxes_s = [6, 6, 6, 6, 6, 6]

    model = SSD_VGG16(num_bboxes_s, 3)
    model.eval()

    input_data = torch.randn(1, 3, 720, 1280, dtype=torch.float, requires_grad=False)
    output_locs, output_clfs = model(input_data)

    model_graph = draw_graph(model, input_size=(1, 3, 720, 1280), expand_nested=True)
    visual_graph = model_graph.visual_graph
    graph_svg = visual_graph.pipe(format="pdf")
    with open("output_vgg16.pdf", "wb") as f:
        f.write(graph_svg)

    print(output_locs.shape)
    print(output_clfs.shape)