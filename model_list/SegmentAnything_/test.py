import build_sam

if __name__=='__main__':
    model = build_sam.build_sam_vit_h().cuda()
    
    data = torch.randn((1, 3, 1024, 1024)).cuda()
    target = torch.randn((1,1,1024,1024)).cuda()
    target[target >= 0.5] = 1; target[target != 1] = 0
    oup,image_embedding = model(data,target)
    print(image_embedding.shape)
