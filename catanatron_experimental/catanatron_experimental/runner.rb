player_features = ["player_features,"]
resource_hand_features = ["resource_hand_features,"]
build_production_features = ["build_production_features(True),"]
map_features = ["tile_features,", "port_features,", "graph_features,"]
game_features = ["game_features,"]

feature_types = {
    "phbgm" => [],
    "phbg" => [*map_features],
    "phb" => [*game_features, *map_features],
    "ph" => [*build_production_features, *game_features, *map_features],
    "p" => [*resource_hand_features, *build_production_features, *game_features, *map_features]
}

all_features = [*player_features, *resource_hand_features, *build_production_features, *game_features, *map_features]

feature_types_array = ["phbgm", "phbg", "phb", "ph", "p"]

discounts = [0.1, 0.5, 0.9]
bench_rounds = 100

bots = ["sarsa"]
bots_tags = ["Z"]
bench_bots = ["VP", "R"]
features_file_path = "../../catanatron_gym/catanatron_gym/features.py"

bots.each_with_index do |bot, bot_index|
    feature_types_array.each do |feature_type|
        features_file_content = File.read(features_file_path)
        feature_types[feature_type].each do |ft|
            features_file_content.gsub!(ft, "##{ft}")
        end
        File.write(features_file_path, features_file_content)

        discounts.each do |discount|
            experiment_name = "#{bot}_#{feature_type}_#{discount}"
            learning_command_line = "python #{bot}_player.py #{experiment_name} #{discount}"
            system(learning_command_line)

            bench_bots.each do |bench_bot|
                benchmark_command_line = "catanatron-play --players=#{bots_tags[bot_index]}:#{experiment_name},#{bench_bot} --num=#{bench_rounds}"
                bench_mv_command_line = "mv ./data/metrics/#{bot}-player/#{experiment_name}/benchmark_metrics.csv ./data/metrics/#{bot}-player/#{experiment_name}/benchmark_metrics_#{bench_bot}.csv"
                system(benchmark_command_line)
                system(bench_mv_command_line)
            end
            rm_tables_command_line = "rm -rf ./data/tables"
            system(rm_tables_command_line)
        end
    end
end

features_file_content = File.read(features_file_path)
all_features.each do |ft|
    features_file_content.gsub!("##{ft}", ft)
end
File.write(features_file_path, features_file_content)