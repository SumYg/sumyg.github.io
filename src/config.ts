export const SITE = {
  website: "https://sumyg.github.io/", // replace this with your deployed domain
  author: "Sum Yeung",
  profile: "https://sumyg.github.io/about",
  desc: "A minimal, responsive and SEO-friendly Astro blog theme.",
  title: "Sum Yeung",
  ogImage: "astropaper-og.jpg",
  lightAndDarkMode: true,
  postPerIndex: 4,
  postPerPage: 4,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true, // show back button in post detail
  editPost: {
    enabled: true,
    text: "Edit page",
    url: "https://github.com/SumYg/sumyg.github.io/edit/main/",
  },
  dynamicOgImage: true,
  dir: "ltr", // "rtl" | "auto"
  lang: "en", // html lang code. Set this empty and default will be "en"
  timezone: "Canada/Pacific", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
} as const;
