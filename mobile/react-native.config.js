module.exports = {
  dependencies: {
    'react-native-vector-icons': {
      platforms: {
        ios: {
          sourceDir: '../node_modules/react-native-vector-icons/Fonts',
          assets: ['../node_modules/react-native-vector-icons/Fonts/*.ttf'],
        },
        android: {
          sourceDir: '../node_modules/react-native-vector-icons/Fonts',
          assets: ['../node_modules/react-native-vector-icons/Fonts/*.ttf'],
        },
      },
    },
    'react-native-sqlite-storage': {
      platforms: {
        android: {
          sourceDir: '../node_modules/react-native-sqlite-storage/platforms/android',
          packageImportPath: 'io.liteglue.SQLitePluginPackage',
        },
      },
    },
  },
  assets: ['./src/assets/fonts/', './src/assets/images/'],
};