def test(name):
    from ..models.resnet18 import SSD_Resnet18
    import torch
    from torchview import draw_graph

    num_bboxes_s = [6, 6, 6, 6, 6, 6]

    model = SSD_Resnet18(num_bboxes_s, 3)
    model.eval()

    input_data = torch.randn(1, 3, 720, 1280, dtype=torch.float, requires_grad=False)
    locs, clfs = model(input_data)

    model_graph = draw_graph(model, input_size=(1, 3, 720, 1280), expand_nested=True)
    visual_graph = model_graph.visual_graph
    graph_svg = visual_graph.pipe(format="pdf")
    with open(f"{name}.pdf", "wb") as f:
        f.write(graph_svg)

    print(locs.shape)
    print(clfs.shape)
