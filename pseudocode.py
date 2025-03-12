#Pseudocode for algorithms mentioned
########################################################################################################################################

# Feature Detection and Matching

# SIFT
# Detect keypoints in images
function SIFT(image):
    grayscale_image = convert_to_grayscale(image)
    
    scale_space = construct_scale_space(grayscale_image)
    keypoints = []
    
    for each scale in scale_space:
        DoG = compute_Difference_of_Gaussian(scale)
        extrema = detect_extrema(DoG)
        refined_extrema = filter_low_contrast(extrema)
        
        for each keypoint in refined_extrema:
            orientation = compute_gradient_orientation(keypoint)
            descriptor = compute_descriptor(keypoint, orientation)
            keypoints.append((keypoint, descriptor))
    
    return keypoints

# Nearest Neighbor
# Match keypoints between images
function ApproximateNearestNeighbors(descriptors1, descriptors2):
    KDTree = build_KDTree(descriptors1)
    matches = []
    
    for descriptor in descriptors2:
        nearest_neighbor = KDTree.find_nearest(descriptor)
        if distance(descriptor, nearest_neighbor) < threshold:
            matches.append((descriptor, nearest_neighbor))
    
    return matches

# RANSAC
# Filter out bad matches and estimate fundamental matrix
function RANSAC(matched_keypoints, iterations, error_threshold):
    best_model = None
    max_inliers = 0
    best_inliers = []
    
    for i in range(iterations):
        sample = random_select(matched_keypoints, 8)
        F = compute_fundamental_matrix(sample)
        
        inliers = []
        for keypoint in matched_keypoints:
            error = compute_reprojection_error(keypoint, F)
            if error < error_threshold:
                inliers.append(keypoint)
        
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_model = F
            best_inliers = inliers
    
    refined_F = recompute_fundamental_matrix(best_inliers)
    return refined_F, best_inliers

# Track Formation
# Organize keypoints into consistent tracks across images
function BuildTracks(image_matches):
    tracks = {}
    
    for each match in image_matches:
        track_id = find_existing_track(tracks, match)
        if track_id is None:
            track_id = create_new_track(tracks)
        
        add_match_to_track(tracks[track_id], match)
    
    return tracks

######################################################################################################################################

#Reconstruction

# Structure from Motion (SfM)
# Estimate camera poses and reconstruct sparse 3D structure
function StructureFromMotion(image_set, feature_tracks):
    selected_pair = choose_best_initial_pair(feature_tracks)
    E = compute_essential_matrix(selected_pair)
    camera_pose1, camera_pose2 = recover_pose(E)

    point_cloud = triangulate(selected_pair, camera_pose1, camera_pose2)
    registered_images = {selected_pair}

    while len(registered_images) < len(image_set):
        next_image = find_next_best_image(registered_images, feature_tracks)
        camera_pose = estimate_pose_PnP(next_image, point_cloud)

        new_points = triangulate(next_image, camera_pose, point_cloud)
        merge_new_points(point_cloud, new_points)
        registered_images.add(next_image)

        bundle_adjustment(point_cloud, registered_images)
    
    return point_cloud, camera_poses

# Multi-View Stereo (MVS)
# Compute depth maps for denser reconstruction
function multi_view_stereo(images):
    point_cloud = []

    for image in images:
        depth_map = estimate_depth(image)
      
        points = generate_3d_points(depth_map, image)

        point_cloud.extend(points)
    
    return point_cloud

# Surface Reconstruction
# Convert point cloud into a mesh model
def surface_reconstruction(point_cloud, images):
    mesh = poisson_surface_reconstruction(point_cloud)
    
    texture_mapped_mesh = apply_texture_mapping(mesh, images)
    
    return texture_mapped_mesh
